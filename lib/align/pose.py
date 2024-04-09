#!/usr/bin/env python3
""" Holds estimated pose information for a faceswap aligned face """
from __future__ import annotations

import logging
import typing as T

import cv2
import numpy as np

from lib.logger import parse_class_init

from .constants import _MEAN_FACE, LandmarkType

logger = logging.getLogger(__name__)

if T.TYPE_CHECKING:
    from .constants import CenteringType


class PoseEstimate():
    """ Estimates pose from a generic 3D head model for the given 2D face landmarks.

    Parameters
    ----------
    landmarks: :class:`numpy.ndarry`
        The original 68 point landmarks aligned to 0.0 - 1.0 range
    landmarks_type: :class:`~LandmarksType`
        The type of landmarks that are generating this face

    References
    ----------
    Head Pose Estimation using OpenCV and Dlib - https://www.learnopencv.com/tag/solvepnp/
    3D Model points - http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
    """
    _logged_once = False

    def __init__(self, landmarks: np.ndarray, landmarks_type: LandmarkType) -> None:
        logger.trace(parse_class_init(locals()))  # type:ignore[attr-defined]
        self._distortion_coefficients = np.zeros((4, 1))  # Assuming no lens distortion
        self._xyz_2d: np.ndarray | None = None

        if landmarks_type != LandmarkType.LM_2D_68:
            self._log_once("Pose estimation is not available for non-68 point landmarks. Pose and "
                           "offset data will all be returned as the incorrect value of '0'")
        self._landmarks_type = landmarks_type
        self._camera_matrix = self._get_camera_matrix()
        self._rotation, self._translation = self._solve_pnp(landmarks)
        self._offset = self._get_offset()
        self._pitch_yaw_roll: tuple[float, float, float] = (0, 0, 0)
        logger.trace("Initialized %s", self.__class__.__name__)  # type:ignore[attr-defined]

    @property
    def xyz_2d(self) -> np.ndarray:
        """ :class:`numpy.ndarray` projected (x, y) coordinates for each x, y, z point at a
        constant distance from adjusted center of the skull (0.5, 0.5) in the 2D space. """
        if self._xyz_2d is None:
            xyz = cv2.projectPoints(np.array([[6., 0., -2.3],
                                              [0., 6., -2.3],
                                              [0., 0., 3.7]]).astype("float32"),
                                    self._rotation,
                                    self._translation,
                                    self._camera_matrix,
                                    self._distortion_coefficients)[0].squeeze()
            self._xyz_2d = xyz - self._offset["head"]
        return self._xyz_2d

    @property
    def offset(self) -> dict[CenteringType, np.ndarray]:
        """ dict: The amount to offset a standard 0.0 - 1.0 umeyama transformation matrix for a
        from the center of the face (between the eyes) or center of the head (middle of skull)
        rather than the nose area. """
        return self._offset

    @property
    def pitch(self) -> float:
        """ float: The pitch of the aligned face in eular angles """
        if not any(self._pitch_yaw_roll):
            self._get_pitch_yaw_roll()
        return self._pitch_yaw_roll[0]

    @property
    def yaw(self) -> float:
        """ float: The yaw of the aligned face in eular angles """
        if not any(self._pitch_yaw_roll):
            self._get_pitch_yaw_roll()
        return self._pitch_yaw_roll[1]

    @property
    def roll(self) -> float:
        """ float: The roll of the aligned face in eular angles """
        if not any(self._pitch_yaw_roll):
            self._get_pitch_yaw_roll()
        return self._pitch_yaw_roll[2]

    @classmethod
    def _log_once(cls, message: str) -> None:
        """ Log a warning about unsupported landmarks if a message has not already been logged """
        if cls._logged_once:
            return
        logger.warning(message)
        cls._logged_once = True

    def _get_pitch_yaw_roll(self) -> None:
        """ Obtain the yaw, roll and pitch from the :attr:`_rotation` in eular angles. """
        proj_matrix = np.zeros((3, 4), dtype="float32")
        proj_matrix[:3, :3] = cv2.Rodrigues(self._rotation)[0]
        euler = cv2.decomposeProjectionMatrix(proj_matrix)[-1]
        self._pitch_yaw_roll = T.cast(tuple[float, float, float], tuple(euler.squeeze()))
        logger.trace("yaw_pitch: %s", self._pitch_yaw_roll)  # type:ignore[attr-defined]

    @classmethod
    def _get_camera_matrix(cls) -> np.ndarray:
        """ Obtain an estimate of the camera matrix based off the original frame dimensions.

        Returns
        -------
        :class:`numpy.ndarray`
            An estimated camera matrix
        """
        focal_length = 4
        camera_matrix = np.array([[focal_length, 0, 0.5],
                                  [0, focal_length, 0.5],
                                  [0, 0, 1]], dtype="double")
        logger.trace("camera_matrix: %s", camera_matrix)  # type:ignore[attr-defined]
        return camera_matrix

    def _solve_pnp(self, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Solve the Perspective-n-Point for the given landmarks.

        Takes 2D landmarks in world space and estimates the rotation and translation vectors
        in 3D space.

        Parameters
        ----------
        landmarks: :class:`numpy.ndarry`
            The original 68 point landmark co-ordinates relating to the original frame

        Returns
        -------
        rotation: :class:`numpy.ndarray`
            The solved rotation vector
        translation: :class:`numpy.ndarray`
            The solved translation vector
        """
        if self._landmarks_type != LandmarkType.LM_2D_68:
            points: np.ndarray = np.empty([])
            rotation = np.array([[0.0], [0.0], [0.0]])
            translation = rotation.copy()
        else:
            points = landmarks[[6, 7, 8, 9, 10, 17, 21, 22, 26, 31, 32, 33, 34,
                                35, 36, 39, 42, 45, 48, 50, 51, 52, 54, 56, 57, 58]]
            _, rotation, translation = cv2.solvePnP(_MEAN_FACE[LandmarkType.LM_3D_26],
                                                    points,
                                                    self._camera_matrix,
                                                    self._distortion_coefficients,
                                                    flags=cv2.SOLVEPNP_ITERATIVE)
        logger.trace("points: %s, rotation: %s, translation: %s",  # type:ignore[attr-defined]
                     points, rotation, translation)
        return rotation, translation

    def _get_offset(self) -> dict[CenteringType, np.ndarray]:
        """ Obtain the offset between the original center of the extracted face to the new center
        of the head in 2D space.

        Returns
        -------
        :class:`numpy.ndarray`
            The x, y offset of the new center from the old center.
        """
        offset: dict[CenteringType, np.ndarray] = {"legacy": np.array([0.0, 0.0])}
        if self._landmarks_type != LandmarkType.LM_2D_68:
            offset["face"] = np.array([0.0, 0.0])
            offset["head"] = np.array([0.0, 0.0])
        else:
            points: dict[T.Literal["face", "head"], tuple[float, ...]] = {"head": (0.0, 0.0, -2.3),
                                                                          "face": (0.0, -1.5, 4.2)}
            for key, pnts in points.items():
                center = cv2.projectPoints(np.array([pnts]).astype("float32"),
                                           self._rotation,
                                           self._translation,
                                           self._camera_matrix,
                                           self._distortion_coefficients)[0].squeeze()
                logger.trace("center %s: %s", key, center)  # type:ignore[attr-defined]
                offset[key] = center - np.array([0.5, 0.5])
        logger.trace("offset: %s", offset)  # type:ignore[attr-defined]
        return offset
