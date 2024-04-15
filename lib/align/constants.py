#!/usr/bin/env python3
""" Constants that are required across faceswap's lib.align package """
from __future__ import annotations

import typing as T
from enum import Enum

import numpy as np

CenteringType = T.Literal["face", "head", "legacy"]

EXTRACT_RATIOS: dict[CenteringType, float] = {"legacy": 0.375, "face": 0.5, "head": 0.625}
"""dict[Literal["legacy", "face", head"] float]: The amount of padding applied to each
centering type when generating aligned faces """


class LandmarkType(Enum):
    """ Enumeration for the landmark types that Faceswap supports """
    LM_2D_4 = 1
    LM_2D_51 = 2
    LM_2D_68 = 3
    LM_3D_26 = 4

    @classmethod
    def from_shape(cls, shape: tuple[int, ...]) -> LandmarkType:
        """ The landmark type for a given shape

        Parameters
        ----------
        shape: tuple[int, ...]
            The shape to get the landmark type for

        Returns
        -------
        Type[LandmarkType]
            The enum for the given shape

        Raises
        ------
        ValueError
            If the requested shape is not valid
        """
        shapes: dict[tuple[int, ...], LandmarkType] = {(4, 2): cls.LM_2D_4,
                                                       (51, 2): cls.LM_2D_51,
                                                       (68, 2): cls.LM_2D_68,
                                                       (26, 3): cls.LM_3D_26}
        if shape not in shapes:
            raise ValueError(f"The given shape {shape} is not valid. Valid shapes: {list(shapes)}")
        return shapes[shape]


_MEAN_FACE: dict[LandmarkType, np.ndarray] = {
    LandmarkType.LM_2D_4: np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),  # Clockwise from TL
    LandmarkType.LM_2D_51: np.array([
        [0.010086, 0.106454], [0.085135, 0.038915], [0.191003, 0.018748], [0.300643, 0.034489],
        [0.403270, 0.077391], [0.596729, 0.077391], [0.699356, 0.034489], [0.808997, 0.018748],
        [0.914864, 0.038915], [0.989913, 0.106454], [0.500000, 0.203352], [0.500000, 0.307009],
        [0.500000, 0.409805], [0.500000, 0.515625], [0.376753, 0.587326], [0.435909, 0.609345],
        [0.500000, 0.628106], [0.564090, 0.609345], [0.623246, 0.587326], [0.131610, 0.216423],
        [0.196995, 0.178758], [0.275698, 0.179852], [0.344479, 0.231733], [0.270791, 0.245099],
        [0.192616, 0.244077], [0.655520, 0.231733], [0.724301, 0.179852], [0.803005, 0.178758],
        [0.868389, 0.216423], [0.807383, 0.244077], [0.729208, 0.245099], [0.264022, 0.780233],
        [0.350858, 0.745405], [0.438731, 0.727388], [0.500000, 0.742578], [0.561268, 0.727388],
        [0.649141, 0.745405], [0.735977, 0.780233], [0.652032, 0.864805], [0.566594, 0.902192],
        [0.500000, 0.909281], [0.433405, 0.902192], [0.347967, 0.864805], [0.300252, 0.784792],
        [0.437969, 0.778746], [0.500000, 0.785343], [0.562030, 0.778746], [0.699747, 0.784792],
        [0.563237, 0.824182], [0.500000, 0.831803], [0.436763, 0.824182]]),
    LandmarkType.LM_3D_26: np.array([
        [4.056931, -11.432347, 1.636229],   # 8 chin LL
        [1.833492, -12.542305, 4.061275],   # 7 chin L
        [0.0, -12.901019, 4.070434],        # 6 chin C
        [-1.833492, -12.542305, 4.061275],  # 5 chin R
        [-4.056931, -11.432347, 1.636229],  # 4 chin RR
        [6.825897, 1.275284, 4.402142],     # 33 L eyebrow L
        [1.330353, 1.636816, 6.903745],     # 29 L eyebrow R
        [-1.330353, 1.636816, 6.903745],    # 34 R eyebrow L
        [-6.825897, 1.275284, 4.402142],    # 38 R eyebrow R
        [1.930245, -5.060977, 5.914376],    # 54 nose LL
        [0.746313, -5.136947, 6.263227],    # 53 nose L
        [0.0, -5.485328, 6.76343],          # 52 nose C
        [-0.746313, -5.136947, 6.263227],   # 51 nose R
        [-1.930245, -5.060977, 5.914376],   # 50 nose RR
        [5.311432, 0.0, 3.987654],          # 13 L eye L
        [1.78993, -0.091703, 4.413414],     # 17 L eye R
        [-1.78993, -0.091703, 4.413414],    # 25 R eye L
        [-5.311432, 0.0, 3.987654],         # 21 R eye R
        [2.774015, -7.566103, 5.048531],    # 43 mouth L
        [0.509714, -7.056507, 6.566167],    # 42 mouth top L
        [0.0, -7.131772, 6.704956],         # 41 mouth top C
        [-0.509714, -7.056507, 6.566167],   # 40 mouth top R
        [-2.774015, -7.566103, 5.048531],   # 39 mouth R
        [-0.589441, -8.443925, 6.109526],   # 46 mouth bottom R
        [0.0, -8.601736, 6.097667],         # 45 mouth bottom C
        [0.589441, -8.443925, 6.109526]])}   # 44 mouth bottom L
"""dict[:class:`~LandmarkType, np.ndarray]: 'Mean' landmark points for various landmark types. Used
for aligning faces """

LANDMARK_PARTS: dict[LandmarkType, dict[str, tuple[int, int, bool]]] = {
            LandmarkType.LM_2D_68: {"mouth_outer": (48, 60, True),
                                    "mouth_inner": (60, 68, True),
                                    "right_eyebrow": (17, 22, False),
                                    "left_eyebrow": (22, 27, False),
                                    "right_eye": (36, 42, True),
                                    "left_eye": (42, 48, True),
                                    "nose": (27, 36, False),
                                    "jaw": (0, 17, False),
                                    "chin": (8, 11, False)},
            LandmarkType.LM_2D_4: {"face": (0, 4, True)}}
"""dict[:class:`LandmarkType`, dict[str, tuple[int, int, bool]]: For each landmark type, stores
the (start index, end index, is polygon) information about each part of the face. """
