#!/usr/bin/env python3
"""Holds estimated pose information for a faceswap aligned face """
from __future__ import annotations

import logging
import typing as T

import cv2
import numpy as np

from lib.logger import parse_class_init
from lib.utils import get_module_objects

from .constants import MEAN_FACE, LandmarkType

logger = logging.getLogger(__name__)

if T.TYPE_CHECKING:
    import numpy.typing as npt
    from .constants import CenteringType


_CORE_LMS = np.array([6, 7, 8, 9, 10, 17, 21, 22, 26, 31, 32, 33, 34,
                      35, 36, 39, 42, 45, 48, 50, 51, 52, 54, 56, 57, 58], dtype="int32")
"""The indices used from 68 point landmarks to align to a 3D head"""

_DISTORTION_COEFFICIENTS = np.zeros((4, 1), dtype="float32")
"""The distortion co-efficient for 3D point estimation (assumes no lens distortion)"""

_MEAN_FACE3D = MEAN_FACE[LandmarkType.LM_3D_26]
"""The (26, 3) 3D landmark points for a "mean" head in 3D normalized space"""

_CENTER_OFFSETS: dict[CenteringType, npt.NDArray[np.float32]] = {
    "legacy": np.array([0.0, 0.0, 0.0], dtype="float32"),
    "head": np.array([0.0, 0.0, -2.3], dtype="float32"),
    "face": np.array([0.0, -1.5, 4.2], dtype="float32")
    }
"""The offsets required to shift the center point of a head in 3D space relative to legacy
centering"""


def get_camera_matrix(focal_length: int = 4) -> np.ndarray:
    """Obtain an estimate of a camera matrix in normalized space

    Parameters
    ----------
    focal_length
        The focal length to obtain the matrix for. Default: 4

    Returns
    -------
    An estimated camera matrix
    """
    focal_length = 4
    camera_matrix = np.array([[focal_length, 0, 0.5],
                              [0, focal_length, 0.5],
                              [0, 0, 1]], dtype="double")
    logger.trace("camera_matrix: %s", camera_matrix)  # type:ignore[attr-defined]
    return camera_matrix


class PoseEstimate():
    """Estimates pose from a generic 3D head model for the given 2D face landmarks.

    Parameters
    ----------
    landmarks
        The original 68 point landmarks aligned to 0.0 - 1.0 range
    landmarks_type
        The type of landmarks that are generating this face

    References
    ----------
    Head Pose Estimation using OpenCV and Dlib - https://www.learnopencv.com/tag/solvepnp/
    3D Model points - http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
    """
    _logged_once = False

    def __init__(self, landmarks: np.ndarray, landmarks_type: LandmarkType) -> None:
        logger.trace(parse_class_init(locals()))  # type:ignore[attr-defined]
        self._xyz_2d: np.ndarray | None = None

        if landmarks_type != LandmarkType.LM_2D_68:
            self._log_once("Pose estimation is not available for non-68 point landmarks. Pose and "
                           "offset data will all be returned as the incorrect value of '0'")
        self._landmarks_type = landmarks_type
        self._camera_matrix = get_camera_matrix()
        self._rotation, self._translation = self._solve_pnp(landmarks)
        self._offset = self._get_offset()
        self._pitch_yaw_roll: tuple[float, float, float] = (0, 0, 0)
        logger.trace("Initialized %s", self.__class__.__name__)  # type:ignore[attr-defined]

    @property
    def xyz_2d(self) -> np.ndarray:
        """projected (x, y) coordinates for each x, y, z point at a constant distance from adjusted
        center of the skull (0.5, 0.5) in the 2D space."""
        if self._xyz_2d is None:
            xyz = cv2.projectPoints(np.array([[6., 0., -2.3],
                                              [0., 6., -2.3],
                                              [0., 0., 3.7]]).astype("float32"),
                                    self._rotation,
                                    self._translation,
                                    self._camera_matrix,
                                    _DISTORTION_COEFFICIENTS)[0].squeeze()
            self._xyz_2d = xyz - self._offset["head"]
        return self._xyz_2d

    @property
    def offset(self) -> dict[CenteringType, np.ndarray]:
        """The amount to offset a standard 0.0 - 1.0 Umeyama transformation matrix from the center
        of the face (between the eyes) or center of the head (middle of skull) rather than the nose
        area."""
        return self._offset

    @property
    def pitch(self) -> float:
        """The pitch of the aligned face in Eular angles"""
        if not any(self._pitch_yaw_roll):
            self._get_pitch_yaw_roll()
        return self._pitch_yaw_roll[0]

    @property
    def yaw(self) -> float:
        """The yaw of the aligned face in Eular angles"""
        if not any(self._pitch_yaw_roll):
            self._get_pitch_yaw_roll()
        return self._pitch_yaw_roll[1]

    @property
    def roll(self) -> float:
        """The roll of the aligned face in Eular angles"""
        if not any(self._pitch_yaw_roll):
            self._get_pitch_yaw_roll()
        return self._pitch_yaw_roll[2]

    @classmethod
    def _log_once(cls, message: str) -> None:
        """Log a warning about unsupported landmarks if a message has not already been logged"""
        if cls._logged_once:
            return
        logger.warning(message)
        cls._logged_once = True

    def _get_pitch_yaw_roll(self) -> None:
        """Obtain the yaw, roll and pitch from the :attr:`_rotation` in Eular angles."""
        proj_matrix = np.zeros((3, 4), dtype="float32")
        proj_matrix[:3, :3] = cv2.Rodrigues(self._rotation)[0]
        euler = cv2.decomposeProjectionMatrix(proj_matrix)[-1]
        self._pitch_yaw_roll = T.cast(tuple[float, float, float], tuple(euler.squeeze()))
        logger.trace("yaw_pitch: %s", self._pitch_yaw_roll)  # type:ignore[attr-defined]

    def _solve_pnp(self, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Solve the Perspective-n-Point for the given landmarks.

        Takes 2D landmarks in world space and estimates the rotation and translation vectors
        in 3D space.

        Parameters
        ----------
        landmarks
            The original 68 point landmark co-ordinates relating to the original frame

        Returns
        -------
        rotation
            The solved rotation vector
        translation
            The solved translation vector
        """
        if self._landmarks_type != LandmarkType.LM_2D_68:
            points: np.ndarray = np.empty([])
            rotation = np.array([[0.0], [0.0], [0.0]])
            translation = rotation.copy()
        else:
            points = landmarks[_CORE_LMS]
            _, rotation, translation = cv2.solvePnP(_MEAN_FACE3D,
                                                    points,
                                                    self._camera_matrix,
                                                    _DISTORTION_COEFFICIENTS,
                                                    flags=cv2.SOLVEPNP_ITERATIVE)
        logger.trace("points: %s, rotation: %s, translation: %s",  # type:ignore[attr-defined]
                     points, rotation, translation)
        return rotation, translation

    def _get_offset(self) -> dict[CenteringType, npt.NDArray[np.float32]]:
        """Obtain the offset between the original center of the extracted face to the new center
        of the head in 2D space.

        Returns
        -------
        The x, y offset of the new center from the old center.
        """
        legacy = np.array([0.0, 0.0], dtype="float32")
        offset: dict[CenteringType, npt.NDArray[np.float32]] = {}
        if self._landmarks_type != LandmarkType.LM_2D_68:
            offset["legacy"] = legacy
            offset["face"] = np.array([0.0, 0.0], dtype="float32")
            offset["head"] = np.array([0.0, 0.0], dtype="float32")
        else:
            for key, points in _CENTER_OFFSETS.items():
                if key == "legacy":
                    offset[key] = legacy
                    continue
                center = cv2.projectPoints(np.array([points]).astype("float32"),
                                           self._rotation,
                                           self._translation,
                                           self._camera_matrix,
                                           _DISTORTION_COEFFICIENTS)[0].squeeze().astype("float32")
                logger.trace("center %s: %s", key, center)  # type:ignore[attr-defined]
                offset[key] = center - np.array([0.5, 0.5], dtype="float32")
        logger.trace("offset: %s", offset)  # type:ignore[attr-defined]
        return offset


class Batch3D:
    """Functions to perform 3D space calculations on batches """
    _camera_matrix = get_camera_matrix()
    _legacy_offset = np.array([[0.0, 0.0]], dtype="float32")
    _to_center_shift = np.array([[0.5, 0.5]], dtype="float32")

    @classmethod
    def solve_pnp(cls, landmarks: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Estimate rotation and translation from a mean 3D head model

        Parameters
        ----------
        landmarks
            The (N, 68, 2) 2D normalized landmark points to obtain the rotation and translation
            vectors for

        Returns
        -------
        The rotation and translation vectors for the given landmarks in format:
        ```
        (rotation, N, 3, 1
         translation, N, 3, 1)
        ```
        """
        core_lms = np.ascontiguousarray(landmarks[:, _CORE_LMS])
        retval = np.array([cv2.solvePnP(_MEAN_FACE3D,
                                        lms,
                                        cls._camera_matrix,
                                        _DISTORTION_COEFFICIENTS,
                                        flags=cv2.SOLVEPNP_ITERATIVE)[1:]
                           for lms in core_lms]).astype("float32").swapaxes(0, 1)
        return retval

    @classmethod
    def rodrigues(cls, vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Perform batch conversion of rotation vectors to rotation matrices

        Parameters
        ----------
        vectors
            The (N, 3, 1) rotation vectors to convert

        Returns
        -------
        The (N, 3, 3) rotation matrices
        """
        vectors = vectors.reshape(-1, 3)
        theta = np.linalg.norm(vectors, axis=1, keepdims=True)
        units = vectors / (theta + 1e-12)

        k = np.zeros((vectors.shape[0], 3, 3), dtype="float32")
        k[:, 0, 1] = -units[:, 2]
        k[:, 0, 2] = units[:, 1]
        k[:, 1, 0] = units[:, 2]
        k[:, 1, 2] = -units[:, 0]
        k[:, 2, 0] = -units[:, 1]
        k[:, 2, 1] = units[:, 0]

        ident = np.eye(3, dtype="float32")
        retval = ident + np.sin(theta)[:, None] * k + (1 - np.cos(theta))[:, None] * (k @ k)
        return retval

    @classmethod
    def project_points(cls,
                       points: npt.NDArray[np.float32],
                       rotation_vectors: npt.NDArray[np.float32],
                       translation_vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Batch protection of points from 3D space to 2D space

        Parameters
        ----------
        points
            The (N, M, 3) points to project
        rotation_vectors
            The (N, 3, 1) rotation vectors for projection
        translation_vectors
            The (N, 3, 1) translation vectors for projection

        Returns
        -------
        The (N, M, 2) projected points in 2D space
        """
        rot = cls.rodrigues(rotation_vectors)
        x_cam = np.einsum('nij,nmj->nmi', rot, points) + translation_vectors.swapaxes(1, 2)
        x_y = x_cam[..., :2] / x_cam[..., 2: 3]

        cam = cls._camera_matrix
        retval = np.empty_like(x_y)
        retval[:, :, 0] = cam[0, 0] * x_y[..., 0] + cam[0, 2]
        retval[:, :, 1] = cam[1, 1] * x_y[..., 1] + cam[1, 2]
        return retval

    @classmethod
    def get_offsets(cls,
                    centering: CenteringType,
                    rotation_vectors: npt.NDArray[np.float32],
                    translation_vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Obtain the offset for moving normalized 68 point landmarks from legacy centering

        Parameters
        ----------
        centering
            The centering type to obtain the offset for
        rotation_vectors
            The (N, 3, 1) batch of rotation vectors to receive offsets for
        translation_vectors
            The (N, 3, 1) batch of translation vectors to receive offsets for

        Returns
        -------
        The (N, 2) offsets for the given rotation/translation vector
        """
        batch_size = rotation_vectors.shape[0]
        if centering == "legacy":
            return np.broadcast_to(cls._legacy_offset, (batch_size, 2))
        points3d = np.broadcast_to(_CENTER_OFFSETS[centering][None], (batch_size, 3))
        offsets = cls.project_points(points3d[:, None, :],
                                     rotation_vectors,
                                     translation_vectors)[:, 0]
        return offsets - cls._to_center_shift


__all__ = get_module_objects(__name__)
