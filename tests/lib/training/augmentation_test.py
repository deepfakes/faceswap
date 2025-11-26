#!/usr/bin python3
""" Pytest unit tests for :mod:`lib.training.augmentation` """
import typing as T

import cv2
import numpy as np
import pytest
import pytest_mock

from lib.config import ConfigValueType
from lib.training.augmentation import (ConstantsAugmentation, ConstantsColor, ConstantsTransform,
                                       ConstantsWarp, ImageAugmentation)

# pylint:disable=protected-access

MODULE_PREFIX = "lib.training.augmentation"
_CONFIG = T.cast(
    dict[str, ConfigValueType],
    {"color_clahe_chance": 50, "color_clahe_max_size": 4, "color_lightness": 30, "color_ab": 8,
     "rotation_range": 10, "zoom_amount": 5, "shift_range": 5, "random_flip": 50})


# CONSTANTS #
_CLAHE_CONF = ((_CONFIG, 64, "valid_64px"),
               ({"color_clahe_chance": 25, "color_clahe_max_size": 8}, 384, "valid_384px"),
               ({"color_clahe_chance": 50.1, "color_clahe_max_size": 4}, 256, "invalid_chance"),
               ({"color_clahe_chance": 50, "color_clahe_max_size": 4.3}, 256, "invalid_size"))


@pytest.mark.parametrize(("config", "size", "status"), _CLAHE_CONF,
                         ids=[x[-1] for x in _CLAHE_CONF])
def test_constants_get_clahe(config: dict[str, T.Any], size: int, status: str) -> None:
    """ Test ConstantsAugmentation._get_clahe works as expected """
    if status.startswith("valid"):
        contrast, chance, max_size = ConstantsAugmentation._get_clahe(config, size)
        assert isinstance(contrast, int)
        assert isinstance(chance, float)
        assert isinstance(max_size, int)
        assert contrast == max(2, size // 128)
        assert chance == config["color_clahe_chance"] / 100.
        assert max_size == config["color_clahe_max_size"]
    else:
        with pytest.raises(AssertionError):
            ConstantsAugmentation._get_clahe(config, size)


_LAB_CONF = ((_CONFIG, "valid"),
             ({"color_lightness": 30.0, "color_ab": 8}, "invalid_l"),
             ({"color_lightness": 30, "color_ab": 8.0}, "invalid_ab"))


@pytest.mark.parametrize(("config", "status"), _LAB_CONF, ids=[x[-1] for x in _LAB_CONF])
def test_constants_get_lab(config: dict[str, T.Any], status: str) -> None:
    """ Test ConstantsAugmentation._get_lab works as expected """
    if status == "valid":
        lab_adjust = ConstantsAugmentation._get_lab(config)
        assert isinstance(lab_adjust, np.ndarray)
        assert lab_adjust.dtype == np.float32
        assert lab_adjust.shape == (3, )
        assert lab_adjust[0] == config["color_lightness"] / 100.
        assert lab_adjust[1] == config["color_ab"] / 100.
        assert lab_adjust[2] == config["color_ab"] / 100.
    else:
        with pytest.raises(AssertionError):
            ConstantsAugmentation._get_lab(config)


_CLAHE_LAB_CONF = (
    (_CONFIG, "valid"),
    ({"color_clahe_chance": 50.0, "color_clahe_max_size": 4.0,
      "color_lightness": 30, "color_ab": 8},
     "invalid_clahe"),
    ({"color_clahe_chance": 50, "color_clahe_max_size": 4,
      "color_lightness": 30.0, "color_ab": 8.0},
     "invalid_lab"))


@pytest.mark.parametrize(("config", "status"), _CLAHE_LAB_CONF,
                         ids=[x[-1] for x in _CLAHE_LAB_CONF])
def test_constants_get_color(config: dict[str, T.Any],
                             status: str,
                             mocker: pytest_mock.MockerFixture) -> None:
    """ Test ConstantsAugmentation._get_color works as expected """
    if status == "valid":
        clahe_mock = mocker.patch(f"{MODULE_PREFIX}.ConstantsAugmentation._get_clahe",
                                  return_value=(1, 2.0, 3))
        lab_mock = mocker.patch(f"{MODULE_PREFIX}.ConstantsAugmentation._get_lab",
                                return_value=np.array([1.0, 2.0, 3.0], dtype="float32"))
        color = ConstantsAugmentation._get_color(config, 256)
        clahe_mock.assert_called_once_with(config, 256)
        lab_mock.assert_called_once_with(config)
        assert isinstance(color, ConstantsColor)
        assert isinstance(color.clahe_base_contrast, int)
        assert isinstance(color.clahe_chance, float)
        assert isinstance(color.clahe_max_size, int)
        assert isinstance(color.lab_adjust, np.ndarray)

        assert color.clahe_base_contrast == clahe_mock.return_value[0]
        assert color.clahe_chance == clahe_mock.return_value[1]
        assert color.clahe_max_size == clahe_mock.return_value[2]
        assert np.all(color.lab_adjust == lab_mock.return_value)
    else:
        with pytest.raises(AssertionError):
            ConstantsAugmentation._get_color(config, 256)


_TRANSFORM_CONF = (
    (_CONFIG, 64, "valid_64px"),
    (_CONFIG, 384, "valid_384px"),
    ({"rotation_range": 10.0, "zoom_amount": 5, "shift_range": 5, "random_flip": 50}, 256,
     "invalid_range"),
    ({"rotation_range": 10, "zoom_amount": 5.0, "shift_range": 5, "random_flip": 50}, 256,
     "invalid_zoom"),
    ({"rotation_range": 10, "zoom_amount": 5, "shift_range": 5.0, "random_flip": 50}, 256,
     "invalid_shift"),
    ({"rotation_range": 10, "zoom_amount": 5, "shift_range": 5.0, "random_flip": 5.0}, 256,
     "invalid_flip"))


@pytest.mark.parametrize(("config", "size", "status"),
                         _TRANSFORM_CONF, ids=[x[-1] for x in _TRANSFORM_CONF])
def test_constants_get_transform(config: dict[str, T.Any], size: int, status: str) -> None:
    """ Test ConstantsAugmentation._get_transform works as expected """
    if status.startswith("valid"):
        transform = ConstantsAugmentation._get_transform(config, size)
        assert isinstance(transform, ConstantsTransform)
        assert isinstance(transform.rotation, int)
        assert isinstance(transform.zoom, float)
        assert isinstance(transform.shift, float)
        assert isinstance(transform.flip, float)
        assert transform.rotation == config["rotation_range"]
        assert transform.zoom == config["zoom_amount"] / 100.
        assert transform.shift == (config["shift_range"] / 100.) * size
        assert transform.flip == config["random_flip"] / 100.
    else:
        with pytest.raises(AssertionError):
            ConstantsAugmentation._get_transform(config, size)


@pytest.mark.parametrize(("size", "batch_size"), ((64, 16), (384, 32)))
def test_constants_get_warp_to_landmarks(size: int, batch_size: int) -> None:
    """ Test ConstantsAugmentation._get_warp_to_landmarks works as expected """
    anchors, grids = ConstantsAugmentation._get_warp_to_landmarks(size, batch_size)
    assert isinstance(anchors, np.ndarray)
    assert isinstance(grids, np.ndarray)

    assert anchors.dtype == np.int32
    assert anchors.shape == (batch_size, 8, 2)
    assert anchors.min() == 0
    assert anchors.max() == size - 1

    assert grids.dtype == np.float32
    assert grids.shape == (2, size, size)
    assert grids.min() == 0.
    assert grids.max() == size - 1


@pytest.mark.parametrize(("size", "batch_size"), ((64, 16), (384, 32)))
def test_constants_get_warp(size: int, batch_size: int, mocker: pytest_mock.MockerFixture) -> None:
    """ Test ConstantsAugmentation._get_warp works as expected """
    warp_lm_mock = mocker.patch(
        f"{MODULE_PREFIX}.ConstantsAugmentation._get_warp_to_landmarks",
        return_value=((np.random.random((batch_size, 8, 2)) * 100).astype("int32"),
                      (np.random.random((2, size, size))).astype("float32")))
    warp_pad = int(1.25 * size)

    warps = ConstantsAugmentation._get_warp(size, batch_size)

    warp_lm_mock.assert_called_once_with(size, batch_size)

    assert isinstance(warps, ConstantsWarp)

    assert isinstance(warps.maps, np.ndarray)
    assert warps.maps.dtype == "float32"
    assert warps.maps.shape == (batch_size, 2, 5, 5)
    assert warps.maps.min() == 0.
    assert warps.maps.mean() == size / 2.
    assert warps.maps.max() == size

    assert isinstance(warps.pad, tuple)
    assert len(warps.pad) == 2
    assert all(isinstance(x, int) for x in warps.pad)
    assert all(x == warp_pad for x in warps.pad)

    assert isinstance(warps.slices, slice)
    assert warps.slices.step is None
    assert warps.slices.start == warp_pad // 10
    assert warps.slices.stop == -warp_pad // 10

    assert isinstance(warps.scale, float)
    assert warps.scale == 5 / 256 * size

    assert isinstance(warps.lm_edge_anchors, np.ndarray)
    assert warps.lm_edge_anchors.dtype == warp_lm_mock.return_value[0].dtype
    assert warps.lm_edge_anchors.shape == warp_lm_mock.return_value[0].shape
    assert np.all(warps.lm_edge_anchors == warp_lm_mock.return_value[0])

    assert isinstance(warps.lm_grids, np.ndarray)
    assert warps.lm_grids.dtype == warp_lm_mock.return_value[1].dtype
    assert warps.lm_grids.shape == warp_lm_mock.return_value[1].shape
    assert np.all(warps.lm_grids == warp_lm_mock.return_value[1])

    assert isinstance(warps.lm_scale, float)
    assert warps.lm_scale == 2 / 256 * size


@pytest.mark.parametrize(("size", "batch_size"), ((64, 16), (384, 32)))
def test_constants_from_config(size: int, batch_size: int, mocker: pytest_mock.MockerFixture
                               ) -> None:
    """ Test that ConstantsAugmentation.from_config executes correctly """
    conf = _CONFIG
    constants = ConstantsAugmentation.from_config(conf, size, batch_size)
    assert isinstance(constants, ConstantsAugmentation)
    assert isinstance(constants.color, ConstantsColor)
    assert isinstance(constants.transform, ConstantsTransform)
    assert isinstance(constants.warp, ConstantsWarp)

    color_mock = mocker.patch(f"{MODULE_PREFIX}.ConstantsAugmentation._get_color")
    transform_mock = mocker.patch(f"{MODULE_PREFIX}.ConstantsAugmentation._get_transform")
    warp_mock = mocker.patch(f"{MODULE_PREFIX}.ConstantsAugmentation._get_warp")
    ConstantsAugmentation.from_config(conf, size, batch_size)
    color_mock.assert_called_once_with(conf, size)
    transform_mock.assert_called_once_with(conf, size)
    warp_mock.assert_called_once_with(size, batch_size)


# IMAGE AUGMENTATION #
def get_batch(batch_size, size: int) -> np.ndarray:
    """ Obtain a batch of random float32 image data for the given batch size and height/width """
    return (np.random.random((batch_size, size, size, 3)) * 255).astype("uint8")


def get_instance(batch_size, size) -> ImageAugmentation:
    """ Obtain an ImageAugmentation instance for the given batch size and size """
    return ImageAugmentation(batch_size, size, _CONFIG)


@pytest.mark.parametrize(("size", "batch_size"), ((64, 16), (384, 32)))
def test_image_augmentation_init(size: int, batch_size: int) -> None:
    """ Test ImageAugmentation initializes """
    attrs = {"_processing_size": int,
             "_batch_size": int,
             "_constants": ConstantsAugmentation}
    instance = get_instance(batch_size, size)

    assert all(x in instance.__dict__ for x in attrs)
    assert all(x in attrs for x in instance.__dict__)
    assert isinstance(instance._batch_size, int)
    assert isinstance(instance._processing_size, int)
    assert isinstance(instance._constants, ConstantsAugmentation)
    assert instance._batch_size == batch_size
    assert instance._processing_size == size


@pytest.mark.parametrize(("size", "batch_size"), ((64, 16), (384, 32)))
def test_image_augmentation_random_lab(size: int,
                                       batch_size: int,
                                       mocker: pytest_mock.MockerFixture) -> None:
    """ Test that ImageAugmentation._random_lab executes as expected """
    batch = get_batch(batch_size, size)
    original = batch.copy()
    instance = get_instance(batch_size, size)

    instance._random_lab(batch)
    assert original.shape == batch.shape
    assert original.dtype == batch.dtype
    assert not np.allclose(original, batch)

    randoms_mock = mocker.patch(f"{MODULE_PREFIX}.np.random.uniform")
    instance._random_lab(batch)
    randoms_mock.assert_called_once()


@pytest.mark.parametrize(("size", "batch_size"), ((64, 16), (384, 32)))
def test_image_augmentation_random_clahe(size: int,
                                         batch_size: int,
                                         mocker: pytest_mock.MockerFixture) -> None:
    """ Test that ImageAugmentation._random_clahe executes as expected """
    # Expected output
    batch = get_batch(batch_size, size)
    original = batch.copy()
    instance = get_instance(batch_size, size)

    instance._random_clahe(batch)
    assert original.shape == batch.shape
    assert original.dtype == batch.dtype
    assert not np.allclose(original, batch)

    # Functions called
    rand_ret = np.random.rand(batch_size)
    rand_mock = mocker.patch(f"{MODULE_PREFIX}.np.random.rand",
                             return_value=rand_ret)

    where_ret = np.where(rand_ret < instance._constants.color.clahe_chance)
    where_mock = mocker.patch(f"{MODULE_PREFIX}.np.where",
                              return_value=where_ret)

    randint_ret = np.random.randint(instance._constants.color.clahe_max_size,
                                    size=where_ret[0].shape[0],
                                    dtype="uint8")
    randint_mock = mocker.patch(f"{MODULE_PREFIX}.np.random.randint",
                                return_value=randint_ret)

    grid_sizes = (randint_ret *
                  (instance._constants.color.clahe_base_contrast //
                   2)) + instance._constants.color.clahe_base_contrast
    clahe_calls = [mocker.call(clipLimit=2.0, tileGridSize=(grid, grid)) for grid in grid_sizes]
    clahe_mock = mocker.patch(f"{MODULE_PREFIX}.cv2.createCLAHE",
                              return_value=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3)))

    batch = get_batch(batch_size, size)
    instance._random_clahe(batch)

    rand_mock.assert_called_once_with(batch_size)
    where_mock.assert_called_once()
    randint_mock.assert_called_once_with(instance._constants.color.clahe_max_size + 1,
                                         size=where_ret[0].shape[0],
                                         dtype="uint8")
    clahe_mock.assert_has_calls(clahe_calls)  # type:ignore


@pytest.mark.parametrize(("size", "batch_size"), ((64, 16), (384, 32)))
def test_image_augmentation_color_adjust(size: int,
                                         batch_size: int,
                                         mocker: pytest_mock.MockerFixture) -> None:
    """ Test that ImageAugmentation._color_adjust executes as expected """
    batch = get_batch(batch_size, size)
    output = get_instance(batch_size, size).color_adjust(batch)
    assert output.shape == batch.shape
    assert output.dtype == batch.dtype
    assert not np.allclose(output, batch)

    batch_convert_mock = mocker.patch(f"{MODULE_PREFIX}.batch_convert_color")
    lab_mock = mocker.patch(f"{MODULE_PREFIX}.ImageAugmentation._random_lab")
    clahe_mock = mocker.patch(f"{MODULE_PREFIX}.ImageAugmentation._random_clahe")

    batch = get_batch(batch_size, size)
    get_instance(batch_size, size).color_adjust(batch)

    assert batch_convert_mock.call_count == 2
    lab_mock.assert_called_once()
    clahe_mock.assert_called_once()


@pytest.mark.parametrize(("size", "batch_size"), ((64, 16), (384, 32)))
def test_image_augmentation_transform(size: int,
                                      batch_size: int,
                                      mocker: pytest_mock.MockerFixture) -> None:
    """ Test that ImageAugmentation.transform executes as expected """
    batch = get_batch(batch_size, size)
    instance = get_instance(batch_size, size)
    original = batch.copy()
    instance.transform(batch)

    assert original.shape == batch.shape
    assert original.dtype == batch.dtype
    assert not np.allclose(original, batch)

    rand_ret = [np.random.uniform(-10, 10, size=batch_size).astype("float32"),
                np.random.uniform(.95, 1.05, size=batch_size).astype("float32"),
                np.random.uniform(-9.2, 9.2, size=(batch_size, 2)).astype("float32")]
    rand_calls = [mocker.call(-instance._constants.transform.rotation,
                              instance._constants.transform.rotation,
                              size=batch_size),
                  mocker.call(1 - instance._constants.transform.zoom,
                              1 + instance._constants.transform.zoom,
                              size=batch_size),
                  mocker.call(-instance._constants.transform.shift,
                              instance._constants.transform.shift,
                              size=(batch_size, 2))]
    rand_mock = mocker.patch(f"{MODULE_PREFIX}.np.random.uniform",
                             side_effect=rand_ret)

    rotmat_mock = mocker.patch(
        f"{MODULE_PREFIX}.cv2.getRotationMatrix2D",
        return_value=np.array([[1.0, 0.0, -2.0], [-1.0, 1.0, 5.0]]).astype("float32"))

    affine_mock = mocker.patch(f"{MODULE_PREFIX}.cv2.warpAffine")

    batch = get_batch(batch_size, size)
    get_instance(batch_size, size).transform(batch)

    rand_mock.assert_has_calls(rand_calls)  # type:ignore
    assert rotmat_mock.call_count == batch_size
    assert affine_mock.call_count == batch_size


@pytest.mark.parametrize(("size", "batch_size"), ((64, 16), (384, 32)))
def test_image_augmentation_random_flip(size: int,
                                        batch_size: int,
                                        mocker: pytest_mock.MockerFixture) -> None:
    """ Test that ImageAugmentation.random_flip executes as expected """
    batch = get_batch(batch_size, size)
    original = batch.copy()
    get_instance(batch_size, size).random_flip(batch)

    assert original.shape == batch.shape
    assert original.dtype == batch.dtype
    assert not np.allclose(original, batch)

    rand_ret = np.random.rand(batch_size)
    rand_mock = mocker.patch(f"{MODULE_PREFIX}.np.random.rand", return_value=rand_ret)
    where_mock = mocker.patch(f"{MODULE_PREFIX}.np.where")

    batch = get_batch(batch_size, size)
    get_instance(batch_size, size).random_flip(batch)

    rand_mock.assert_called_once_with(batch_size)
    where_mock.assert_called_once()


@pytest.mark.parametrize(("size", "batch_size"), ((64, 16), (384, 32)))
def test_image_augmentation_random_warp(size: int,
                                        batch_size: int,
                                        mocker: pytest_mock.MockerFixture) -> None:
    """ Test that ImageAugmentation._random_warp executes as expected """
    batch = get_batch(batch_size, size)
    instance = get_instance(batch_size, size)
    output = instance._random_warp(batch)

    assert output.shape == batch.shape
    assert output.dtype == batch.dtype
    assert not np.allclose(output, batch)

    rand_ret = np.random.normal(size=(batch_size, 2, 5, 5), scale=0.02).astype("float32")
    rand_mock = mocker.patch(f"{MODULE_PREFIX}.np.random.normal", return_value=rand_ret)

    eval_ret = np.ones_like(rand_ret)
    eval_mock = mocker.patch(f"{MODULE_PREFIX}.ne.evaluate", return_value=eval_ret)

    resize_ret = np.ones((size, size)).astype("float32")
    resize_mock = mocker.patch(f"{MODULE_PREFIX}.cv2.resize", return_value=resize_ret)

    remap_mock = mocker.patch(f"{MODULE_PREFIX}.cv2.remap")

    instance._random_warp(batch)

    rand_mock.assert_called_once_with(size=(batch_size, 2, 5, 5),
                                      scale=instance._constants.warp.scale)
    eval_mock.assert_called_once()
    assert resize_mock.call_count == batch_size * 2
    assert remap_mock.call_count == batch_size


@pytest.mark.parametrize(("size", "batch_size"), ((64, 16), (384, 32)))
def test_image_augmentation_random_warp_landmarks(size: int,
                                                  batch_size: int,
                                                  mocker: pytest_mock.MockerFixture) -> None:
    """ Test that ImageAugmentation._random_warp_landmarks executes as expected """
    src_points = np.random.random(size=(batch_size, 68, 2)).astype("float32") * size
    dst_points = np.random.random(size=(batch_size, 68, 2)).astype("float32") * size

    batch = get_batch(batch_size, size)
    instance = get_instance(batch_size, size)
    output = instance._random_warp_landmarks(batch, src_points, dst_points)

    assert output.shape == batch.shape
    assert output.dtype == batch.dtype
    assert not np.allclose(output, batch)

    rand_ret = np.random.normal(size=dst_points.shape, scale=0.01)
    rand_mock = mocker.patch(f"{MODULE_PREFIX}.np.random.normal", return_value=rand_ret)

    hull_ret = [cv2.convexHull(np.concatenate([src[17:], dst[17:]], axis=0))
                for src, dst in zip(src_points.astype("int32"),
                                    (dst_points + rand_ret).astype("int32"))]
    hull_mock = mocker.patch(f"{MODULE_PREFIX}.cv2.convexHull", side_effect=hull_ret)

    remap_mock = mocker.patch(f"{MODULE_PREFIX}.cv2.remap")

    instance._random_warp_landmarks(batch, src_points, dst_points)

    rand_mock.assert_called_once_with(size=(dst_points.shape),
                                      scale=instance._constants.warp.lm_scale)
    assert hull_mock.call_count == batch_size
    assert remap_mock.call_count == batch_size


@pytest.mark.parametrize(("size", "batch_size", "to_landmarks"),
                         ((64, 16, True), (384, 32, False)))
def test_image_augmentation_warp(size: int,
                                 batch_size: int,
                                 to_landmarks: bool,
                                 mocker: pytest_mock.MockerFixture) -> None:
    """ Test that ImageAugmentation.warp executes as expected """
    kwargs = {}
    if to_landmarks:
        kwargs["batch_src_points"] = np.random.random(
            size=(batch_size, 68, 2)).astype("float32") * size
        kwargs["batch_dst_points"] = np.random.random(
            size=(batch_size, 68, 2)).astype("float32") * size
    batch = get_batch(batch_size, size)
    output = get_instance(batch_size, size).warp(batch, to_landmarks, **kwargs)

    assert output.shape == batch.shape
    assert output.dtype == batch.dtype
    assert not np.allclose(output, batch)

    if to_landmarks:
        with pytest.raises(AssertionError):
            get_instance(batch_size, size).warp(batch,
                                                to_landmarks,
                                                batch_src_points=kwargs["batch_src_points"],
                                                batch_dst_points=None)
        with pytest.raises(AssertionError):
            get_instance(batch_size, size).warp(batch,
                                                to_landmarks,
                                                batch_src_points=None,
                                                batch_dst_points=kwargs["batch_dst_points"])

    warp_mock = mocker.patch(f"{MODULE_PREFIX}.ImageAugmentation._random_warp")
    warp_lm_mock = mocker.patch(f"{MODULE_PREFIX}.ImageAugmentation._random_warp_landmarks")

    get_instance(batch_size, size).warp(batch, to_landmarks, **kwargs)
    if to_landmarks:
        warp_mock.assert_not_called()
        warp_lm_mock.assert_called_once()
    else:
        warp_mock.assert_called_once()
        warp_lm_mock.assert_not_called()
