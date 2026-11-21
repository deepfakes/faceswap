#!/usr/bin/env python3
"""Tests for :mod:`lib.infer.objects`"""
import numpy as np

from lib.infer.objects import ExtractBatchMask, FrameFaces


def test_detected_faces_skips_empty_masks():
    """FrameFaces.detected_faces should skip empty mask arrays without raising IndexError."""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    bboxes = np.array([[10, 10, 40, 40]], dtype=np.int32)
    valid_mask = ExtractBatchMask(
        centering="face",
        matrices=np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32),
        storage_size=32,
        masks=np.full((1, 32, 32), 255, dtype=np.uint8),
    )
    empty_mask = ExtractBatchMask(
        centering="face",
        matrices=np.empty((0, 2, 3), dtype=np.float32),
        storage_size=32,
        masks=np.empty((0, 32, 32), dtype=np.uint8),
    )

    media = FrameFaces(
        "frame_0011.png",
        image,
        bboxes=bboxes,
        masks={"bisenet-fp": valid_mask, "custom": empty_mask},
    )

    faces = media.detected_faces
    assert len(faces) == 1
    assert "bisenet-fp" in faces[0].mask
    assert "custom" not in faces[0].mask


def test_detected_faces_skips_short_masks_for_extra_faces():
    """FrameFaces.detected_faces should not raise when a mask batch has fewer entries than faces."""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    bboxes = np.array([[5, 5, 25, 25], [30, 30, 55, 55]], dtype=np.int32)
    partial_mask = ExtractBatchMask(
        centering="face",
        matrices=np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=np.float32),
        storage_size=32,
        masks=np.full((1, 32, 32), 255, dtype=np.uint8),
    )

    media = FrameFaces(
        "frame_0012.png",
        image,
        bboxes=bboxes,
        masks={"custom": partial_mask},
    )

    faces = media.detected_faces
    assert len(faces) == 2
    assert "custom" in faces[0].mask
    assert "custom" not in faces[1].mask
