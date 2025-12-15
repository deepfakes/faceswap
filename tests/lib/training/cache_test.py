#!/usr/bin python3
""" Pytest unit tests for :mod:`lib.training.cache` """
import os
import typing as T

from threading import Lock

import numpy as np
import pytest
import pytest_mock

from lib.align.constants import LandmarkType
from lib.training import cache as cache_mod
from lib.utils import FaceswapError
from plugins.train import train_config as cfg


from tests.lib.config.helpers import patch_config  # # pylint:disable=unused-import  # noqa[F401]

# pylint:disable=protected-access,invalid-name,redefined-outer-name


# ## HELPERS ###

MODULE_PREFIX = "lib.training.cache"
_DUMMY_IMAGE_LIST = ["/path/to/img1.png", "~/img2.png", "img3.png"]


def _get_config(centering="face", vertical_offset=0):
    """ Return a fresh valid config """
    return {"centering": centering,
            "vertical_offset": vertical_offset}


STANDARD_CACHE_ARGS = (_DUMMY_IMAGE_LIST, 256, 1.0)
STANDARD_MASK_ARGS = (256, 1.0, "face")


# ## MASK PROCESSING ###

def get_mask_config(penalized_mask_loss=True,
                    learn_mask=True,
                    mask_type="extended",
                    mask_dilation=1.0,
                    mask_kernel=3,
                    mask_threshold=4,
                    mask_eye_multiplier=2,
                    mask_mouth_multiplier=3):
    """ Generate the mask config dictionary with the given arguments """
    return {"penalized_mask_loss": penalized_mask_loss,
            "learn_mask": learn_mask,
            "mask_type": mask_type,
            "mask_dilation": mask_dilation,
            "mask_blur_kernel": mask_kernel,
            "mask_threshold": mask_threshold,
            "eye_multiplier": mask_eye_multiplier,
            "mouth_multiplier": mask_mouth_multiplier}


_MASK_CONFIG_PARAMS = (
    (get_mask_config(True, True, "extended", 1.0, 3, 4, 2, 3), "pass-penalize|learn"),
    (get_mask_config(True, False, "components", 0.0, 5, 4, 1, 2), "pass-penalize"),
    (get_mask_config(False, True, "custom", -2.0, 6, 1, 3, 1), "pass-learn"),
    (get_mask_config(True, True, None, 1.0, 6, 1, 3, 2), "pass-mask-disable1"),
    (get_mask_config(False, False, "extended", 1.0, 6, 1, 3, 2), "pass-mask-disable2"),
    (get_mask_config(True, True, "extended", 1.0, 1, 3, 1, 1), "pass-multiplier-disable"),
    (get_mask_config("Error", True, "extended", 1.0, 1, 3, 2, 3), "fail-penalize"),
    (get_mask_config(True, 1.4, "extended", 1.0, 1, 3, 2, 3), "fail-learn"),
    (get_mask_config(True, True, 999, 1.0, 1, 3, 2, 3), "fail-type"),
    (get_mask_config(True, True, "extended", 23, 1, 3, 2, 3), "fail-dilation"),
    (get_mask_config(True, True, "extended", 1.0, 1.2, 3, 2, 3), "fail-kernel"),
    (get_mask_config(True, True, "extended", 1.0, 1, "fail", 2, 3), "fail-threshold"),
    (get_mask_config(True, True, "extended", 1.0, 1, 3, 3.9, 3), "fail-eye-multi"),
    (get_mask_config(True, True, "extended", 1.0, 1, 3, 2, "fail"), "fail-mouth-multi"))
_MASK_CONFIG_IDS = [x[-1] for x in _MASK_CONFIG_PARAMS]


@pytest.mark.parametrize(("config", "status"), _MASK_CONFIG_PARAMS, ids=_MASK_CONFIG_IDS)
def test_MaskConfig(config: dict[str, T.Any],
                    status: str,
                    patch_config) -> None:  # noqa[F811]
    """ Test that cache._MaskConfig dataclass initializes from config """
    patch_config(cfg.Loss, config)
    retval = cache_mod._MaskConfig()
    if status.startswith("pass-mask-disable"):
        assert not retval.mask_enabled
    else:
        assert retval.mask_enabled

    if status == "pass-multiplier-disable" or not config["penalized_mask_loss"]:
        assert not retval.multiplier_enabled
    else:
        assert retval.multiplier_enabled


_MASK_INIT_PARAMS = ((64, 0.5, "face", "pass"),
                     (128, 0.75, "head", "pass"),
                     (384, 1.0, "legacy", "pass"),
                     (69.42, 0.75, "head", "fail-size"),
                     (128, "fail", "head", "fail-coverage"),
                     (128, 0.75, "fail", "fail-centering"))
_MASK_INIT_IDS = [x[-1] for x in _MASK_INIT_PARAMS]


@pytest.mark.parametrize(("size", "coverage", "centering", "status"),
                         _MASK_INIT_PARAMS, ids=_MASK_INIT_IDS)
def test_MaskProcessing_init(size,
                             coverage,
                             centering,
                             status: str,
                             mocker: pytest_mock.MockerFixture) -> None:
    """ Test cache._MaskProcessing correctly initializes """
    mock_maskconfig = mocker.MagicMock()
    mocker.patch(f"{MODULE_PREFIX}._MaskConfig", new=mock_maskconfig)

    if not status == "pass":
        with pytest.raises(AssertionError):
            cache_mod._MaskProcessing(size, coverage, centering)
        return

    instance = cache_mod._MaskProcessing(size, coverage, centering)
    attrs = {"_size": int,
             "_coverage": float,
             "_centering": str,
             "_config": mocker.MagicMock}  # Our mocked _MaskConfig

    for attr, dtype in attrs.items():
        assert attr in instance.__dict__
        assert isinstance(instance.__dict__[attr], dtype)
    assert all(x in attrs for x in instance.__dict__)

    assert instance._size == size
    assert instance._coverage == coverage
    assert instance._centering == centering
    mock_maskconfig.assert_called_once()


def test_MaskProcessing_check_mask_exists(mocker: pytest_mock.MockerFixture) -> None:
    """ Test cache._MaskProcessing._check_mask_exists functions as expected """
    mock_det_face = mocker.MagicMock()
    mock_det_face.mask = ["extended", "components"]

    instance = cache_mod._MaskProcessing(*STANDARD_MASK_ARGS)  # type:ignore[arg-type]

    instance._check_mask_exists("", mock_det_face)

    mock_det_face.mask = []
    with pytest.raises(FaceswapError):
        instance._check_mask_exists("", mock_det_face)


@pytest.mark.parametrize(("dilation", "kernel", "threshold"),
                         ((1.0, 3, 4), (-2.5, 5, 2), (3.3, 7, 9)))
def test_MaskProcessing_preprocess(dilation: float,
                                   kernel: int,
                                   threshold: int,
                                   mocker: pytest_mock.MockerFixture,
                                   patch_config) -> None:  # noqa[F811]
    """ Test cache._MaskProcessing._preprocess functions as expected """
    mock_mask = mocker.MagicMock()
    mock_det_face = mocker.MagicMock()
    mock_det_face.mask = {"extended": mock_mask}

    patch_config(cfg.Loss, get_mask_config(mask_dilation=dilation,
                                           mask_kernel=kernel,
                                           mask_threshold=threshold))

    instance = cache_mod._MaskProcessing(*STANDARD_MASK_ARGS)  # type:ignore[arg-type]
    instance._preprocess(mock_det_face, "extended")
    mock_mask.set_dilation.assert_called_once_with(dilation)
    mock_mask.set_blur_and_threshold.assert_called_once_with(blur_kernel=kernel,
                                                             threshold=threshold)


@pytest.mark.parametrize(
        ("mask_centering", "train_centering", "coverage", "y_offset", "size", "mask_size"),
        (("face", "legacy", 0.75, 0.0, 256, 64),
         ("legacy", "head", 0.66, -0.25, 128, 128),
         ("head", "face", 1.0, 0.33, 64, 256)))
def test_MaskProcessing_crop_and_resize(mask_centering: str,  # pylint:disable=too-many-locals
                                        train_centering: T.Literal["legacy", "face", "head"],
                                        coverage: float,
                                        y_offset: float,
                                        size: int,
                                        mask_size: int,
                                        mocker: pytest_mock.MockerFixture) -> None:
    """ Test cache._MaskProcessing._crop_and_resize functions as expected """
    mock_pose = mocker.MagicMock()
    mock_pose.offset = {"face": "face_centering",
                        "legacy": "legacy_centering",
                        "head": "head_centering"}

    mock_det_face = mocker.MagicMock()
    mock_det_face.aligned.pose = mock_pose
    mock_det_face.aligned.y_offset = y_offset

    mock_face_mask = mocker.MagicMock()
    mock_face_mask.__get_item__ = mock_face_mask
    mock_face_mask.shape = (mask_size, mask_size)

    mock_mask = mocker.MagicMock()
    mock_mask.stored_centering = mask_centering
    mock_mask.stored_size = mask_size
    mock_mask.mask = mock_face_mask

    mock_cv2_resize_result = mocker.MagicMock()
    mock_cv2_resize_item = mocker.MagicMock()
    mock_cv2_resize = mocker.patch(f"{MODULE_PREFIX}.cv2.resize",
                                   return_value=mock_cv2_resize_result)
    mock_cv2_resize_result.__getitem__.return_value = mock_cv2_resize_item

    mock_cv2_cubic = mocker.patch(f"{MODULE_PREFIX}.cv2.INTER_CUBIC")
    mock_cv2_area = mocker.patch(f"{MODULE_PREFIX}.cv2.INTER_AREA")

    instance = cache_mod._MaskProcessing(size, coverage, train_centering)

    retval = instance._crop_and_resize(mock_det_face, mock_mask)
    mock_mask.set_sub_crop.assert_called_once_with(mock_pose.offset[mask_centering],
                                                   mock_pose.offset[train_centering],
                                                   train_centering,
                                                   coverage,
                                                   y_offset)
    if mask_size == size:
        assert retval is mock_face_mask
        mock_cv2_resize.assert_not_called()
        return

    assert retval is mock_cv2_resize_item
    interp_used = mock_cv2_cubic if mask_size < size else mock_cv2_area
    mock_cv2_resize.assert_called_once_with(mock_face_mask,
                                            (size, size),
                                            interpolation=interp_used)


@pytest.mark.parametrize("mask_type", (None, "extended", "components"))
def test_MaskProcessing_get_face_mask(mask_type: str | None,
                                      mocker: pytest_mock.MockerFixture,
                                      patch_config) -> None:  # noqa[F811]
    """ Test cache._MaskProcessing._get_face_mask functions as expected """
    patch_config(cfg, _get_config())
    patch_config(cfg.Loss, get_mask_config(mask_type=mask_type))
    instance = cache_mod._MaskProcessing(*STANDARD_MASK_ARGS)  # type:ignore[arg-type]
    assert instance._config.mask_type == mask_type  # sanity check

    instance._check_mask_exists = mocker.MagicMock()  # type:ignore[method-assign]
    preprocess_return = "test_preprocess_return"
    instance._preprocess = mocker.MagicMock(  # type:ignore[method-assign]
        return_value="test_preprocess_return")
    crop_and_resize_return = mocker.MagicMock()
    crop_and_resize_return.shape = (256, 256, 1)
    instance._crop_and_resize = mocker.MagicMock(  # type:ignore[method-assign]
        return_value=crop_and_resize_return)

    filename = "test_filename"
    detected_face = "test_detected_face"

    if mask_type is None:  # Mask disabled
        assert not instance._config.mask_enabled
        retval1 = instance._get_face_mask(filename, detected_face)  # type:ignore[arg-type]
        assert retval1 is None
        instance._check_mask_exists.assert_not_called()  # type:ignore[attr-defined]
        instance._preprocess.assert_not_called()  # type:ignore[attr-defined]
        instance._crop_and_resize.assert_not_called()  # type:ignore[attr-defined]
    else:  # Mask enabled
        assert instance._config.mask_enabled
        retval2 = instance._get_face_mask(filename, detected_face)  # type:ignore[arg-type]
        assert retval2 is crop_and_resize_return
        instance._check_mask_exists.assert_called_once_with(  # type:ignore[attr-defined]
            filename, detected_face)

        instance._preprocess.assert_called_once_with(  # type:ignore[attr-defined]
            detected_face, instance._config.mask_type)

        instance._crop_and_resize.assert_called_once_with(  # type:ignore[attr-defined]
            detected_face, preprocess_return)


@pytest.mark.parametrize(("eye_multiplier", "mouth_multiplier", "size", "enabled"),
                         ((0, 0, 64, False),
                          (1, 1, 64, False),
                          (1, 2, 64, True),
                          (2, 1, 96, True),
                          (2, 3, 128, True),
                          (3, 1, 256, True)))
def test_MaskProcessing_get_localized_mask(eye_multiplier: int,
                                           mouth_multiplier: int,
                                           size: int,
                                           enabled: bool,
                                           mocker: pytest_mock.MockerFixture,
                                           patch_config) -> None:  # noqa[F811]
    """ Test cache._MaskProcessing._get_localized_mask functions as expected """
    args = STANDARD_MASK_ARGS[1:]
    patch_config(cfg.Loss, get_mask_config(mask_eye_multiplier=eye_multiplier,
                                           mask_mouth_multiplier=mouth_multiplier))
    instance = cache_mod._MaskProcessing(size, *args)  # type:ignore[arg-type]

    filename = "filename"
    detected_face = mocker.MagicMock()
    landmark_mask_return_value = mocker.MagicMock()

    detected_face.get_landmark_mask = mocker.MagicMock(return_value=landmark_mask_return_value)

    for area in ("mouth", "eye"):
        retval = instance._get_localized_mask(filename, detected_face, area)
        if not enabled:
            assert retval is None
            detected_face.get_landmark_mask.assert_not_called()
        else:
            assert retval is landmark_mask_return_value

        if enabled:
            detected_face.get_landmark_mask.assert_called_with(area, size // 16, 2.5)
    if enabled:
        assert detected_face.get_landmark_mask.call_count == 2


def test_MaskProcessing_call(mocker: pytest_mock.MockerFixture) -> None:
    """ Test cache._MaskProcessing.__call__ functions as expected """
    instance = cache_mod._MaskProcessing(*STANDARD_MASK_ARGS)  # type:ignore[arg-type]
    face_return = "face_mask"
    area_return = "area_mask"
    instance._get_face_mask = mocker.MagicMock(  # type:ignore[method-assign]
        return_value=face_return)  # type:ignore[method-assign]
    instance._get_localized_mask = mocker.MagicMock(  # type:ignore[method-assign]
        return_value=area_return)  # type:ignore[method-assign]

    filename = "test_filename"
    detected_face = mocker.MagicMock()
    detected_face.store_training_masks = mocker.MagicMock()

    instance(filename, detected_face)

    instance._get_face_mask.assert_called_once_with(  # type:ignore[attr-defined]
        filename, detected_face)

    expected_localized_calls = [mocker.call(filename, detected_face, "eye"),
                                mocker.call(filename, detected_face, "mouth")]
    instance._get_localized_mask.assert_has_calls(  # type:ignore[attr-defined]
        expected_localized_calls, any_order=False)  # pyright:ignore[reportArgumentType]
    assert instance._get_localized_mask.call_count == 2  # type:ignore[attr-defined]

    detected_face.store_training_masks.assert_called_once_with(
        [face_return, area_return, area_return],
        delete_masks=True)


# ## CACHE PROCESSING ###

@pytest.fixture
def face_cache_reset_scenario(mocker: pytest_mock.MockerFixture,
                              request: pytest.FixtureRequest):
    """ Build a scenario for cache._check_reset.

    request.param = {"caches": dict(Literal["a", "b"], bool],
                     "side": Literal["a", "b"]}

    If the key "a" or "b" exist in the caches dict, then that cache exists in the mocked
    cache._FACE_CACHES with a mock representing the return value of the cache.Cache.check_reset()
    value as given

    The mocked Cache item for the currently testing side is returned, or a default mocked item if
    the given side is not meant to be in the _FACE_CACHES dict
    """
    cache_dict = {}
    for side, val in request.param["caches"].items():
        check_mock = mocker.MagicMock()
        check_mock.check_reset.return_value = val
        cache_dict[side] = check_mock
    mocker.patch(f"{MODULE_PREFIX}._FACE_CACHES", new=cache_dict)
    return cache_dict.get(request.param["side"], mocker.MagicMock())


_RESET_PARAMS = [({"side": side, "caches": caches}, expected, f"{name}-{side}")
                 for side in ("a", "b")
                 for caches, expected, name in [
                    ({}, False, "no-cache"),
                    ({"a": False}, False, "a-exists"),
                    ({"b": False}, False, "b-exists"),
                    ({"a": True, "b": False}, side == "b", "a-reset"),
                    ({"a": False, "b": True}, side == "a",  "b-reset"),
                    ({"a": True, "b": True}, True, "both-reset"),
                    ({"a": False, "b": False}, False, "no-reset")]]
_RESET_IDS = [x[-1] for x in _RESET_PARAMS]
_RESET_PARAMS = [x[:-1] for x in _RESET_PARAMS]  # type:ignore[misc]


@pytest.mark.parametrize(("face_cache_reset_scenario", "expected"),
                         _RESET_PARAMS,
                         ids=_RESET_IDS,
                         indirect=["face_cache_reset_scenario"])
def test_check_reset(face_cache_reset_scenario, expected):  # pylint:disable=redefined-outer-name
    """ Test that cache._check_reset functions as expected """
    this_cache = face_cache_reset_scenario
    assert cache_mod._check_reset(this_cache) == expected


@pytest.mark.parametrize(
        ("filenames", "size", "coverage_ratio", "centering"),
        [(_DUMMY_IMAGE_LIST, 256, 1.0, "face"),
         (_DUMMY_IMAGE_LIST[:-1], 96, .75, "head"),
         (_DUMMY_IMAGE_LIST[2:], 384, .66, "legacy")])
def test_Cache_init(filenames, size, coverage_ratio, centering, patch_config):  # noqa[F811]
    """ Test that cache.Cache correctly initializes """
    attrs = {"_lock": type(Lock()),
             "_cache_info": dict,
             "_config": cache_mod._CacheConfig,
             "_partially_loaded": list,
             "_image_count": int,
             "_cache": dict,
             "_aligned_landmarks": dict,
             "_extract_version": float,
             "_mask_prepare": cache_mod._MaskProcessing}
    patch_config(cfg, _get_config(centering=centering))
    instance = cache_mod.Cache(filenames, size, coverage_ratio)

    for attr, attr_type in attrs.items():
        assert attr in instance.__dict__
        assert isinstance(getattr(instance, attr), attr_type)
    for key in instance.__dict__:
        assert key in attrs

    assert set(instance._cache_info) == {"cache_full", "has_reset"}
    assert all(x is False for x in instance._cache_info.values())

    assert not instance._partially_loaded
    assert not instance._cache
    assert instance._image_count == len(filenames)
    assert not instance._aligned_landmarks
    assert instance._extract_version == 0.0
    assert instance._config.size == size
    assert instance._config.centering == centering
    assert instance._config.coverage == coverage_ratio


def test_Cache_cache_full(mocker: pytest_mock.MockerFixture):
    """ Test that cache.Cache.cache_full property behaves correctly """
    instance = cache_mod.Cache(*STANDARD_CACHE_ARGS)
    instance._lock = mocker.MagicMock()

    is_full1 = instance.cache_full
    assert not is_full1
    instance._lock.__enter__.assert_called_once()  # type:ignore[attr-defined]
    instance._lock.__exit__.assert_called_once()  # type:ignore[attr-defined]

    instance._cache_info["cache_full"] = True
    is_full2 = instance.cache_full
    assert is_full2
    # lock not called when cache is full
    instance._lock.__enter__.assert_called_once()  # type:ignore[attr-defined]
    instance._lock.__exit__.assert_called_once()  # type:ignore[attr-defined]


def test_Cache_aligned_landmarks(mocker: pytest_mock.MockerFixture):
    """ Test that cache.Cache.aligned_landmarks property behaves correcly """
    instance = cache_mod.Cache(*STANDARD_CACHE_ARGS)
    instance._lock = mocker.MagicMock()
    for fname in _DUMMY_IMAGE_LIST:
        mock_face = mocker.MagicMock()
        mock_face.aligned.landmarks = f"landmarks_for_{fname}"
        instance._cache[fname] = mock_face

    retval1 = instance.aligned_landmarks
    assert len(_DUMMY_IMAGE_LIST) == len(retval1)
    assert retval1 == {fname: f"landmarks_for_{fname}" for fname in _DUMMY_IMAGE_LIST}
    instance._lock.__enter__.assert_called_once()  # type:ignore[attr-defined]
    instance._lock.__exit__.assert_called_once()  # type:ignore[attr-defined]

    retval2 = instance.aligned_landmarks
    assert len(_DUMMY_IMAGE_LIST) == len(retval1)
    assert retval2 == {fname: f"landmarks_for_{fname}" for fname in _DUMMY_IMAGE_LIST}
    # lock not called after first call has populated
    instance._lock.__enter__.assert_called_once()  # type:ignore[attr-defined]
    instance._lock.__exit__.assert_called_once()  # type:ignore[attr-defined]


@pytest.mark.parametrize("size", (64, 96, 128, 256, 384))
def test_Cache_size(size):
    """ Test that cache.Cache.size property returns correctly """
    instance = cache_mod.Cache(_DUMMY_IMAGE_LIST, size, 1.0)
    assert instance.size == size


def test_Cache_check_reset():
    """ Test that cache.Cache.check_reset behaves correctly """
    instance = cache_mod.Cache(*STANDARD_CACHE_ARGS)
    retval1 = instance.check_reset()
    assert not retval1
    assert not instance._cache_info["has_reset"]

    instance._cache_info["has_reset"] = True
    retval2 = instance.check_reset()
    assert retval2
    assert not instance._cache_info["has_reset"]


@pytest.mark.parametrize("filenames",
                         (_DUMMY_IMAGE_LIST, _DUMMY_IMAGE_LIST[:-1], _DUMMY_IMAGE_LIST[2:]))
def test_Cache_get_items(filenames: list[str]) -> None:
    """ Test that cache.Cache.get_items returns correctly """
    instance = cache_mod.Cache(filenames, 256, 1.0)
    instance._cache = {os.path.basename(f): f"faces_for_{f}"  # type:ignore[misc]
                       for f in filenames}

    retval = instance.get_items(filenames)
    assert retval == [f"faces_for_{f}" for f in filenames]


@pytest.mark.parametrize("set_flag", (True, False), ids=("set-flag", "no-set-flag"))
def test_Cache_reset_cache(set_flag: bool,
                           mocker: pytest_mock.MockerFixture,
                           patch_config) -> None:  # noqa[F811]
    """ Test that cache.Cache._reset_cache functions correctly """
    patch_config(cfg, _get_config(centering="head"))
    mock_warn = mocker.MagicMock()
    mocker.patch(f"{MODULE_PREFIX}.logger.warning", mock_warn)
    instance = cache_mod.Cache(*STANDARD_CACHE_ARGS)
    instance._cache = {"test": "cache"}  # type:ignore[dict-item]
    instance._cache_info["cache_full"] = True

    assert instance._config.centering != "legacy"
    assert instance._cache
    assert instance._cache_info["cache_full"]

    instance._reset_cache(set_flag)

    assert instance._config.centering == "legacy"
    assert not instance._cache
    assert instance._cache_info["cache_full"] is False

    if set_flag:
        mock_warn.assert_called_once()


@pytest.mark.parametrize("png_meta",
                         ({"source": {"alignments_version": 1.0}},
                          {"source": {"alignments_version": 2.0}},
                          {"source": {"alignments_version": 2.2}}),
                         ids=("v1.0", "v2.0", "v2.2"))
def test_Cache_validate_version(png_meta, mocker):
    """ Test that cache.Cache._validate_version executes correctly """
    instance = cache_mod.Cache(*STANDARD_CACHE_ARGS)
    instance._reset_cache = mocker.MagicMock()
    fname = "test_filename.png"
    version = png_meta["source"]["alignments_version"]

    if version == 1.0:
        for centering in ("legacy", "face"):
            instance._extract_version = 0.0
            instance._config.centering = centering
            instance._validate_version(png_meta, fname)
            if centering == "legacy":
                instance._reset_cache.assert_not_called()
            else:
                instance._reset_cache.assert_called_once_with(True)
            assert instance._extract_version == version
    else:
        instance._validate_version(png_meta, fname)
        instance._reset_cache.assert_not_called()
        assert instance._extract_version == version

    instance._extract_version = 1.0  # Legacy alignments have been seen
    if version > 1.0:  # Newer alignments inbound
        with pytest.raises(FaceswapError):
            instance._validate_version(png_meta, fname)
    else:
        instance._validate_version(png_meta, fname)

    instance._extract_version = 2.0  # Newer alignments have been seen
    if version < 2.0:  # Legacy alignments inbound
        with pytest.raises(FaceswapError):
            instance._validate_version(png_meta, fname)
        return  # Exit early on 1.0 because cannot pass any more tests

    instance._validate_version(png_meta, fname)
    if version > 2.0:
        assert instance._extract_version == 2.0  # Defaulted to lowest version

    instance._extract_version = 2.5
    instance._validate_version(png_meta, fname)
    assert instance._extract_version == version  # Defaulted to lowest version


_DET_FACE_PARAMS = ((64, 0.5, 0, 1.0),
                    (96, 0.75, 1, 1.0),
                    (256, 0.66, 2, 2.0),
                    (384, 1.0, 3.0, 2.2))
_DET_FACE_IDS = [f"size:{x[0]}|coverage:{x[1]}|y-offset:{x[2]}|extract-vers:{x[3]}"
                 for x in _DET_FACE_PARAMS]


@pytest.mark.parametrize(("size", "coverage", "y_offset", "extract_version"),
                         _DET_FACE_PARAMS,
                         ids=_DET_FACE_IDS)
def test_Cache_load_detected_face(size: int,
                                  coverage: float,
                                  y_offset: int | float,
                                  extract_version: float,
                                  mocker: pytest_mock.MockerFixture,
                                  patch_config) -> None:  # noqa[F811]
    """ Test that cache.Cache._load_detected_faces executes correctly """
    patch_config(cfg, _get_config(vertical_offset=y_offset))
    instance = cache_mod.Cache(_DUMMY_IMAGE_LIST, size, coverage)
    instance._extract_version = extract_version
    alignments = {}  # type:ignore[var-annotated]

    mock_det_face = mocker.MagicMock()
    mock_det_face.from_png_meta = mocker.MagicMock()
    mock_det_face.load_aligned = mocker.MagicMock()
    mocker.patch(f"{MODULE_PREFIX}.DetectedFace", return_value=mock_det_face)

    retval = instance._load_detected_face("", alignments)  # type:ignore[arg-type]
    assert retval is mock_det_face
    mock_det_face.from_png_meta.assert_called_once_with(alignments)
    mock_det_face.load_aligned.assert_called_once_with(None,
                                                       size=instance._config.size,
                                                       centering=instance._config.centering,
                                                       coverage_ratio=instance._config.coverage,
                                                       y_offset=y_offset / 100.,
                                                       is_aligned=True,
                                                       is_legacy=extract_version == 1.0)


@pytest.mark.parametrize("partially_loaded", (True, False), ids=("partial", "full"))
def test_Cache_populate_cache(partially_loaded: bool,
                              mocker: pytest_mock.MockerFixture) -> None:
    """ Test that cache.Cache._populate_cache executes correctly """
    already_cached = ["/path/to/img4.png", "/path/img5.png"]
    needs_cache = _DUMMY_IMAGE_LIST
    filenames = _DUMMY_IMAGE_LIST + already_cached
    metadata = [{"alignments": f"{f}_alignments"} for f in filenames]

    instance = cache_mod.Cache(*STANDARD_CACHE_ARGS)
    instance._validate_version = mocker.MagicMock()  # type:ignore[method-assign]
    instance._mask_prepare = mocker.MagicMock()
    instance._cache = {os.path.basename(f): "existing"  # type:ignore[misc]
                       for f in filenames if f not in needs_cache}

    mock_detected_faces = {f: mocker.MagicMock() for f in needs_cache}

    if partially_loaded:
        instance._cache.update({os.path.basename(f): mock_detected_faces[f] for f in needs_cache})
        instance._partially_loaded = [os.path.basename(f) for f in filenames]  # Add our partials
    else:
        instance._load_detected_face = mocker.MagicMock(  # type:ignore[method-assign]
            side_effect=[mock_detected_faces[f] for f in needs_cache])

    # Call the function
    instance._populate_cache(needs_cache, metadata, filenames)  # type:ignore[arg-type]

    expected_validate = [mocker.call(metadata[idx], f) for idx, f in enumerate(needs_cache)]
    instance._validate_version.assert_has_calls(expected_validate,  # type:ignore[attr-defined]
                                                any_order=False)
    assert instance._validate_version.call_count == len(needs_cache)  # type:ignore[attr-defined]

    expected_mask_prepare = [mocker.call(f, mock_detected_faces[f]) for f in needs_cache]
    instance._mask_prepare.assert_has_calls(expected_mask_prepare,  # type:ignore[attr-defined]
                                            any_order=False)
    assert instance._mask_prepare.call_count == len(needs_cache)  # type:ignore[attr-defined]

    assert len(instance._cache) == len(filenames)
    for filename in filenames:
        key = os.path.basename(filename)
        assert key in instance._cache
        if filename in needs_cache:  # item got added/updated
            assert instance._cache[key] == mock_detected_faces[filename]
        else:  # item pre-existed
            assert instance._cache[key] == "existing"

    if partially_loaded:
        assert instance._partially_loaded == [os.path.basename(f) for f in filenames
                                              if f not in needs_cache]


@pytest.mark.parametrize("scenario", ("read-error", "size-error", "success"))
def test_Cache_get_batch_with_metadata(scenario: str, mocker: pytest_mock.MockerFixture) -> None:
    """ Test that cache.Cache._get_batch_with_metadata executes correctly """
    instance = cache_mod.Cache(*STANDARD_CACHE_ARGS)
    filenames = ["list", "of", "test", "filenames"]

    mock_read_image_batch = mocker.MagicMock()
    if scenario == "read-error":
        mock_read_image_batch.side_effect = ValueError("inhomogeneous")
    else:
        mock_return = (mocker.MagicMock(), {"test": "meta"})
        if scenario == "size-error":
            mock_return[0].shape = (len(filenames), )
        else:
            mock_return[0].shape = (len(filenames), 64, 64, 3)
        mock_read_image_batch.return_value = mock_return

    mocker.patch(f"{MODULE_PREFIX}.read_image_batch", new=mock_read_image_batch)

    if scenario != "success":
        with pytest.raises(FaceswapError):
            instance._get_batch_with_metadata(filenames)
        mock_read_image_batch.assert_called_once_with(filenames, with_metadata=True)
        return

    retval = instance._get_batch_with_metadata(filenames)
    mock_read_image_batch.assert_called_once_with(filenames, with_metadata=True)
    assert retval == mock_return  # pyright:ignore[reportPossiblyUnboundVariable]


@pytest.mark.parametrize("scenario", ("full", "not-full", "partial"))
def test_Cache_update_cache_full(scenario: bool, mocker: pytest_mock.MockerFixture) -> None:
    """ Test that cache.Cache._update_cache_full executes correctly """
    mock_verbose = mocker.patch(f"{MODULE_PREFIX}.logger.verbose")
    filenames = ["test", "file", "names"]
    instance = cache_mod.Cache(*STANDARD_CACHE_ARGS)
    instance._image_count = 10

    assert instance._cache_info["cache_full"] is False
    assert not instance._cache
    assert not instance._partially_loaded

    if scenario == "full":
        instance._cache = {i: i for i in range(10)}  # type:ignore[misc]
    if scenario == "patial":
        instance._cache = {i: i for i in range(10)}  # type:ignore[misc]
        instance._partially_loaded = filenames.copy()

    instance._update_cache_full(filenames)

    if scenario == "full":
        assert instance._cache_info["cache_full"] is True
        mock_verbose.assert_called_once()
    else:
        assert instance._cache_info["cache_full"] is False
        mock_verbose.assert_not_called()


@pytest.mark.parametrize("scenario", ("full", "partial", "empty", "needs-reset"))
def test_Cache_cache_metadata(scenario: str, mocker: pytest_mock.MockerFixture) -> None:
    """ Test that cache.Cache.cache_metadata executes correctly """
    mock_check_reset = mocker.patch(f"{MODULE_PREFIX}._check_reset")
    mock_check_reset.return_value = scenario == "needs-reset"
    mock_return_batch = mocker.MagicMock()

    mock_read_image_batch = mocker.patch(f"{MODULE_PREFIX}.read_image_batch",
                                         return_value=mock_return_batch)

    instance = cache_mod.Cache(*STANDARD_CACHE_ARGS)
    filenames = _DUMMY_IMAGE_LIST.copy()

    if scenario in ("full", "partial"):
        instance._cache = {os.path.basename(f): f for f in filenames}  # type:ignore[misc]
    if scenario == "partial":
        instance._partially_loaded = [os.path.basename(f) for f in filenames]

    instance._lock = mocker.MagicMock()
    instance._reset_cache = mocker.MagicMock()  # type:ignore[method-assign]
    returned_meta = {"test": "meta"}
    instance._get_batch_with_metadata = mocker.MagicMock(  # type:ignore[method-assign]
        return_value=(mock_return_batch, returned_meta))
    instance._populate_cache = mocker.MagicMock()  # type:ignore[method-assign]
    instance._update_cache_full = mocker.MagicMock()  # type:ignore[method-assign]

    retval = instance.cache_metadata(filenames)  # Call

    instance._lock.__enter__.assert_called_once()  # type:ignore[attr-defined]
    instance._lock.__exit__.assert_called_once()  # type:ignore[attr-defined]

    mock_check_reset.assert_called_once_with(instance)

    if scenario == "needs-reset":
        instance._reset_cache.assert_called_once_with(False)  # type:ignore[attr-defined]
    else:
        instance._reset_cache.assert_not_called()  # type:ignore[attr-defined]

    if scenario == "full":
        mock_read_image_batch.assert_called_once_with(filenames)
        instance._get_batch_with_metadata.assert_not_called()  # type:ignore[attr-defined]
        instance._populate_cache.assert_not_called()  # type:ignore[attr-defined]
        instance._update_cache_full.assert_not_called()  # type:ignore[attr-defined]
    else:
        mock_read_image_batch.assert_not_called()
        instance._get_batch_with_metadata.assert_called_once_with(  # type:ignore[attr-defined]
            filenames)
        instance._populate_cache.assert_called_once_with(  # type:ignore[attr-defined]
            filenames, returned_meta, filenames)
        instance._update_cache_full.assert_called_once_with(filenames)  # type:ignore[attr-defined]

    assert retval is mock_return_batch


@pytest.mark.parametrize("scenario", ("fail-meta", "fail-landmarks", "success"))
def test_Cache_pre_fill(scenario: str, mocker: pytest_mock.MockerFixture) -> None:
    """ Test that cache.Cache.prefill executes correctly """
    filenames = _DUMMY_IMAGE_LIST.copy()
    mock_read_image_batch = mocker.patch(f"{MODULE_PREFIX}.read_image_meta_batch")
    side_effect_read_image_batch = [(f, {}) for f in filenames]  # type:ignore[var-annotated]
    if scenario != "fail-meta":  # Set successful return data
        for effect in side_effect_read_image_batch:
            effect[1]["itxt"] = {"alignments": [1, 2, 3]}
    mock_read_image_batch.side_effect = [side_effect_read_image_batch]

    instance = cache_mod.Cache(*STANDARD_CACHE_ARGS)
    instance._lock = mocker.MagicMock()
    instance._validate_version = mocker.MagicMock()  # type:ignore[method-assign]
    mock_detected_faces = [mocker.MagicMock() for _ in filenames]

    for m in mock_detected_faces:
        m.aligned.landmark_type = (LandmarkType.LM_2D_68 if scenario == "success" else "fail")
    instance._load_detected_face = mocker.MagicMock(  # type:ignore[method-assign]
        side_effect=mock_detected_faces)

    if scenario in ("fail-meta", "fail-landmarks"):
        with pytest.raises(FaceswapError):
            instance.pre_fill(filenames, "a")
        instance._lock.__enter__.assert_called_once()  # type:ignore[attr-defined]
        instance._lock.__exit__.assert_called_once()  # type:ignore[attr-defined]
        mock_read_image_batch.assert_called_once_with(filenames)
        if scenario == "fail-meta":
            instance._validate_version.assert_not_called()  # type:ignore[attr-defined]
            instance._load_detected_face.assert_not_called()  # type:ignore[attr-defined]
        else:
            meta = side_effect_read_image_batch[0][1]["itxt"]
            instance._validate_version.assert_called_once_with(  # type:ignore[attr-defined]
                meta, filenames[0])
            instance._load_detected_face.assert_called_once_with(  # type:ignore[attr-defined]
                filenames[0], meta["alignments"])
        return

    # success
    instance.pre_fill(filenames, "a")
    instance._lock.__enter__.assert_called_once()  # type:ignore[attr-defined]
    instance._lock.__exit__.assert_called_once()  # type:ignore[attr-defined]
    mock_read_image_batch.assert_called_once_with(filenames)

    fname_calls = [x[0] for x in side_effect_read_image_batch]
    meta_calls = [x[1]["itxt"] for x in side_effect_read_image_batch]
    call_validate = [mocker.call(l, f) for f, l in zip(fname_calls, meta_calls)]
    call_det_face = [mocker.call(f, l["alignments"]) for f, l in zip(fname_calls, meta_calls)]

    instance._validate_version.assert_has_calls(  # type:ignore[attr-defined]
        call_validate, any_order=False)  # type:ignore[attr-defined]
    assert instance._validate_version.call_count == len(filenames)  # type:ignore[attr-defined]
    instance._load_detected_face.assert_has_calls(  # type:ignore[attr-defined]
        call_det_face, any_order=False)  # type:ignore[attr-defined]
    assert instance._load_detected_face.call_count == len(filenames)  # type:ignore[attr-defined]

    assert instance._cache == {os.path.basename(f): d for f, d in zip(filenames,
                                                                      mock_detected_faces)}
    assert instance._partially_loaded == [os.path.basename(f) for f in filenames]


_PARAMS_GET = (("a", _DUMMY_IMAGE_LIST, 256, 1.),
               ("b", _DUMMY_IMAGE_LIST, 256, 1.),
               ("c", _DUMMY_IMAGE_LIST, 256, 1.),
               ("a", None, 256, 1,),
               ("a", _DUMMY_IMAGE_LIST, None, 1.),
               ("a", _DUMMY_IMAGE_LIST, 256, None))
_IDS_GET = ("pass-a", "pass-b", "fail-side", "fail-no-filenames",
            "fail-no-size", "fail-no-coverage")


@pytest.mark.parametrize(("side", "filenames", "size", "coverage_ratio", "status"),
                         (x + (y,) for x, y in zip(_PARAMS_GET, _IDS_GET)),
                         ids=_IDS_GET)
def test_get_cache_initial(side: str,
                           filenames: list[str],
                           size: int,
                           coverage_ratio: float,
                           status: str,
                           mocker: pytest_mock.MockerFixture) -> None:
    """ Test cache.get_cache function when the cache does not yet exist """
    mocker.patch(f"{MODULE_PREFIX}._FACE_CACHES", new={})
    patched_cache = mocker.patch(f"{MODULE_PREFIX}.Cache")
    if status.startswith("fail"):
        with pytest.raises(AssertionError):
            cache_mod.get_cache(side, filenames, size, coverage_ratio)  # type:ignore[arg-type]
        patched_cache.assert_not_called()
        return

    retval = cache_mod.get_cache(side, filenames, size, coverage_ratio)  # type:ignore[arg-type]
    assert side in cache_mod._FACE_CACHES
    patched_cache.assert_called_once_with(filenames, size, coverage_ratio)
    assert cache_mod._FACE_CACHES[side] is patched_cache.return_value
    assert retval is patched_cache.return_value

    retval2 = cache_mod.get_cache(side, filenames, size, coverage_ratio)  # type:ignore[arg-type]
    patched_cache.assert_called_once()  # Not called again
    assert retval2 is retval


_IDS_GET2 = ("pass-a", "pass-b", "fail-side", "pass-no-filenames",
             "pass-no-size", "pass-no-coverage")


@pytest.mark.parametrize(("side", "filenames", "size", "coverage_ratio", "status"),
                         (x + (y,) for x, y in zip(_PARAMS_GET, _IDS_GET2)),
                         ids=_IDS_GET2)
def test_get_cache_exists(side: str,
                          filenames: list[str],
                          size: int,
                          coverage_ratio: float,
                          status: str,
                          mocker: pytest_mock.MockerFixture) -> None:
    """ Test cache.get_cache function when the cache exists """
    mocker.patch(f"{MODULE_PREFIX}._FACE_CACHES", new={"a": mocker.MagicMock(),
                                                       "b": mocker.MagicMock()})
    patched_cache = mocker.patch(f"{MODULE_PREFIX}.Cache")

    if status.startswith("fail"):
        with pytest.raises(AssertionError):
            cache_mod.get_cache(side, filenames, size, coverage_ratio)  # type:ignore[arg-type]
        patched_cache.assert_not_called()
        return

    retval = cache_mod.get_cache(side, filenames, size, coverage_ratio)  # type:ignore[arg-type]
    patched_cache.assert_not_called()
    assert retval is cache_mod._FACE_CACHES[side]


# ## Ring Buffer ## #

_RING_BUFFER_PARAMS = ((2, (384, 384, 3), 2, "uint8"),
                       (16, (128, 128, 3), 5, "float32"),
                       (32, (64, 64, 3), 4, "int32"))
_RING_BUFFER_IDS = [f"bs{x[0]}|{x[1][0]}px|buffer-size{x[2]}|dtype-{x[3]}"
                    for x in _RING_BUFFER_PARAMS]


@pytest.mark.parametrize(("batch_size", "image_shape", "buffer_size", "dtype"),
                         ((2, (384, 384, 3), 2, "uint8"),
                          (16, (128, 128, 3), 5, "float32"),
                          (32, (64, 64, 3), 4, "int32")),
                         ids=_RING_BUFFER_IDS)
def test_RingBuffer_init(batch_size, image_shape, buffer_size, dtype):
    """ test cache.RingBuffer initializes correctly """
    attrs = {"_max_index": int, "_index": int, "_buffer": list}
    instance = cache_mod.RingBuffer(batch_size, image_shape, buffer_size, dtype)

    for attr, attr_type in attrs.items():
        assert attr in instance.__dict__
        assert isinstance(getattr(instance, attr), attr_type)
    for key in instance.__dict__:
        assert key in attrs

    assert instance._max_index == buffer_size - 1
    assert instance._index == 0
    assert len(instance._buffer) == buffer_size
    assert all(isinstance(b, np.ndarray) for b in instance._buffer)
    assert all(b.shape == (batch_size, *image_shape) for b in instance._buffer)
    assert all(b.dtype == dtype for b in instance._buffer)


@pytest.mark.parametrize(("batch_size", "image_shape", "buffer_size", "dtype"),
                         ((2, (384, 384, 3), 2, "uint8"),
                          (16, (128, 128, 3), 5, "float32"),
                          (32, (64, 64, 3), 4, "int32")),
                         ids=_RING_BUFFER_IDS)
def test_RingBuffer_call(batch_size, image_shape, buffer_size, dtype):
    """ Test calling cache.RingBuffer works correctly """
    instance = cache_mod.RingBuffer(batch_size, image_shape, buffer_size, dtype)
    for i in range(buffer_size * 3):
        retval = instance()
        assert isinstance(retval, np.ndarray)
        assert retval.shape == (batch_size, *image_shape)
        assert retval.dtype == dtype
        if i % buffer_size == buffer_size - 1:
            assert instance._index == 0
        else:
            assert instance._index == i % buffer_size + 1
