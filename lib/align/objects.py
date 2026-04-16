#! /usr/env/bin/python3
"""Dataclass objects for holding and serializing alignments data"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, MISSING
import types
import typing as T

import numpy as np
import numpy.typing as npt

from lib.logger import format_array

from .constants import CenteringType


@dataclass
class DataclassDict:
    """Parent DataClass that has methods for loading to and from a dict for data serialization"""
    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                params[k] = format_array(v)
                continue
            if isinstance(v, bytes):
                params[k] = f"{len(v)}b"
                continue
            params[k] = v
        s_params = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"

    @classmethod
    def _object_to_serial(cls, obj: T.Any) -> T.Any:
        """Convert object lists or DataclassDicts serializable items

        Parameters
        ----------
        obj
            The object to convert

        Returns
        -------
        The converted object or original object if not to be converted
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, DataclassDict):
            return obj.to_dict()
        return obj

    def to_dict(self) -> dict[str, T.Any]:
        """Obtain the contents of the dataclass object as a python dictionary

        Returns
        -------
        The dataclass object as a python dictionary, with numpy arrays converted to lists
        """
        retval: dict[str, T.Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (list, tuple)):
                retval[k] = [self._object_to_serial(x) for x in v]
            elif isinstance(v, dict):
                retval[k] = {x: self._object_to_serial(y) for x, y in v.items()}
            else:
                retval[k] = self._object_to_serial(v)
        return retval

    @classmethod
    def _convert_dtype(cls, data_type: T.Any, val: T.Any) -> DataclassDict | np.ndarray | None:
        """Convert a serialized dict to a DataclassDict or list to a numpy array of the correct
        dtype

        Parameters
        ----------
        field_type
            The field type for the incoming value
        value
            The list to convert to a numpy array or DataclassDict

        Returns
        -------
        The inbound item to a DataclassDict or numpy array. ``None`` if the item does not convert
        """
        if isinstance(data_type, type) and issubclass(data_type, DataclassDict):
            return data_type.from_dict(val)

        origin = T.get_origin(data_type)
        if origin is types.UnionType:
            args = tuple(a for a in T.get_args(data_type) if a is not types.NoneType)
            assert len(args) == 1
            if val is None:
                return val
            data_type = args[0]
            origin = T.get_origin(data_type)

        if origin is not np.ndarray:
            return None

        args = T.get_args(data_type)
        dtype = T.get_args(args[1])[0]
        return np.array(val, dtype=dtype)

    @classmethod
    def _parse_dict(cls, field_type: T.Any, value: dict[str, T.Any]) -> dict[str, T.Any]:
        """Parse incoming serialized dicts into their correct nested objects

        Parameters
        ----------
        field_type
            The field type for the incoming value
        value
            The dictionary to parse

        Returns
        -------
        The dictionary with its values converted to the correct datatype
        """
        assert T.get_origin(field_type) is dict
        dtype = T.get_args(field_type)[1]
        retval = {}
        for k, v in value.items():
            converted = cls._convert_dtype(dtype, v)
            if converted is not None:
                retval[k] = converted
                continue
            retval[k] = v
        return retval

    @classmethod
    def _parse_list(cls, field_type: T.Any, value: list[T.Any] | tuple[T.Any]
                    ) -> list[T.Any] | tuple[T.Any]:
        """Parse incoming serialized lists into their correct nested objects

        Parameters
        ----------
        field_type
            The field type for the incoming value
        value
            The list to parse

        Returns
        -------
        The list with its values converted to the correct datatype
        """
        origin = T.get_origin(field_type)
        assert origin in (list, tuple), (
            f"value: {type(value)} field: {T.get_origin(field_type)}")
        dtype = T.get_args(field_type)[0]
        items = []
        for v in value:
            converted = cls._convert_dtype(dtype, v)
            if converted is not None:
                items.append(converted)
                continue
            items.append(v)
        retval = T.cast(list[T.Any] | tuple[T.Any], tuple(items) if origin is tuple else items)
        return retval

    @classmethod
    def from_dict(cls, data_dict: dict[str, T.Any]) -> T.Self:
        """Load the contents from a serialized python dict into this dataclass

        Parameters
        ----------
        data_dict
            The data to load into the dataclass
        """
        inbound = set(data_dict)
        all_fields = set(f.name for f in fields(cls))
        required = set(f.name for f in fields(cls)
                       if f.default is MISSING and f.default_factory is MISSING)
        if not inbound.issubset(all_fields):
            raise ValueError(f"Dictionary keys {sorted(inbound)} should be a subset of dataclass "
                             f"params {sorted(all_fields)}")
        if not required.issubset(inbound):
            raise ValueError(f"Dataclass params {sorted(required)} should be a subset of "
                             f"dictionary keys {sorted(inbound)}")
        type_hints = T.get_type_hints(cls)
        kwargs: dict[str, T.Any] = {}
        for f in fields(cls):
            if f.name not in data_dict:
                continue
            field_type = type_hints.get(f.name)
            val = data_dict[f.name]
            converted = cls._convert_dtype(field_type, val)
            if converted is not None:
                kwargs[f.name] = converted
                continue
            if isinstance(val, dict):
                kwargs[f.name] = cls._parse_dict(field_type, val)
                continue
            if isinstance(val, (list, tuple)):
                kwargs[f.name] = cls._parse_list(field_type, val)
                continue
            kwargs[f.name] = val
        return cls(**kwargs)


@dataclass(repr=False)
class MaskAlignmentsFile(DataclassDict):
    """Dataclass for storing Masks in alignments files and PNG Headers"""
    mask: bytes
    """The zlib compressed UINT8 mask of shape (stored_size, stored_size)"""
    affine_matrix: npt.NDArray[np.float32]
    """The affine matrix that takes the mask from stored space to frame space"""
    interpolator: int
    """The interpolator required to take the mask from stored space to frame space"""
    stored_size: int
    """The size the mask is stored at"""
    stored_centering: CenteringType
    """The (legacy, face, head) centering type of the mask"""


@dataclass(repr=False)
class PNGAlignments(DataclassDict):
    """Base Dataclass for storing a single faces' Alignment Information in Alignments files and PNG
    Headers."""
    x: int
    """The left most point of the bounding box"""
    y: int
    """The top most point of the bounding box"""
    w: int
    """The width of the bounding box"""
    h: int
    """The height of the bounding box"""
    landmarks_xy: npt.NDArray[np.float32]
    """The (x, y) landmark points of the face"""
    mask: dict[str, MaskAlignmentsFile] = field(default_factory=dict)
    """The masks stored for the face"""
    identity: dict[str, npt.NDArray[np.float32]] = field(default_factory=dict)
    """The identity vectors stored for the face"""

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params: dict[str, T.Any] = {}
        for k, v in self.__dict__.items():
            if k in ("landmarks_xy", "thumb"):
                params[k] = None if v is None else f"{type(v)}[{len(v)}]"
                continue
            if k == "identity":
                params[k] = {n: f"{type(i)}[{len(i)}]" for n, i in v.items()}
                continue
            params[k] = v
        s_params = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params})"


@dataclass(repr=False)
class PNGSource(DataclassDict):
    """Dataclass for storing additional meta information in PNG headers."""
    alignments_version: float
    """The alignments file version that created the alignments data"""
    original_filename: str
    """The original filename that this face was saved with"""
    face_index: int
    """The index of this face within the frame"""
    source_filename: str
    """The filename of the original frame the face was extracted from"""
    source_is_video: bool
    """``True`` if the face was extracted from a video. ``False`` if from an image"""
    source_frame_dims: tuple[int, int]
    """The (Height, Width) dimensions of the original frame the face was extracted from"""


@dataclass(repr=False)
class PNGHeader(DataclassDict):
    """Dataclass for storing all alignment and meta information in PNG Headers."""
    alignments: PNGAlignments
    """The alignment information for the face"""
    source: PNGSource
    """The frame source information for the face"""


@dataclass(repr=False)
class FileAlignments(PNGAlignments):
    """Dataclass that holds the same information as PNGAlignments as well as a thumbnail for a
    single face"""
    thumb: npt.NDArray[np.uint8] | None = None
    """96px JPEG thumbnail of the aligned face image stored as a list"""


@dataclass(repr=False)
class AlignmentsEntry(DataclassDict):
    """Holds the alignments entry for a single frame in the Alignments data dictionary"""
    faces: list[FileAlignments] = field(default_factory=list)
    """The detected faces in a frame"""
    video_meta: dict[T.Literal["pts_time", "keyframe"], int] = field(default_factory=dict)
    """The keyframe to pts timestamp mapping for video data"""
