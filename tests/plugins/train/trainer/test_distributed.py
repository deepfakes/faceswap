#!/usr/bin python3
""" Pytest unit tests for :mod:`plugins.train.trainer.distributed` Trainer plug in """
# pylint:disable=protected-access, invalid-name, duplicate-code, too-many-locals

import numpy as np
import pytest
import pytest_mock
import torch

from lib.training.data.collate import BatchMeta
from plugins.train.trainer import distributed as mod_distributed
from plugins.train.trainer import original as mod_original
from plugins.train.trainer import base as mod_base


_MODULE_PREFIX = "plugins.train.trainer.distributed"


class DummyLoss:  # pylint:disable=too-few-public-methods
    """Dummy loss return"""
    total = 1.0


@pytest.mark.parametrize("batch_size", (4, 8, 16, 32, 64))
@pytest.mark.parametrize("outputs", (1, 2, 4))
def test_WrappedModel(batch_size, outputs, mocker):
    """ Test that the wrapped model calls predictions and loss """
    model = mocker.MagicMock()
    instance = mod_distributed.WrappedModel(model)
    assert instance._keras_model is model

    loss_return = DummyLoss()
    model.loss_func = mocker.MagicMock(return_value=loss_return)

    test_dims = (batch_size, 16, 16, 3)

    inp_a = torch.from_numpy(np.random.random(test_dims))
    inp_b = torch.from_numpy(np.random.random(test_dims))
    targets = [torch.from_numpy(np.random.random(test_dims))
               for _ in range(outputs * 2)]
    predictions = [*torch.from_numpy(np.random.random((outputs * 2, *test_dims)))]

    model.return_value = predictions

    # Call forwards
    instance.forward([inp_a, inp_b], targets, BatchMeta().__dict__)

    # Confirm model was called once forward with correct args
    model.assert_called_once()
    model_args, model_kwargs = model.call_args
    assert model_kwargs == {"training": True}
    assert len(model_args) == 1
    assert len(model_args[0]) == 2
    for real, expected in zip(model_args[0], [inp_a, inp_b]):
        assert np.allclose(real.numpy(), expected.numpy())

    # Confirm loss functions correctly called
    assert model.loss_func.call_count == 2


@pytest.fixture
def _trainer_mocked(mocker: pytest_mock.MockFixture):  # noqa:[F811]
    """ Generate a mocked model and feeder object and patch torch GPU count """

    def _apply_patch(gpus=2, batch_size=8):
        patched_cuda_device = mocker.patch(f"{_MODULE_PREFIX}.torch.cuda.device_count")
        patched_cuda_device.return_value = gpus
        patched_parallel = mocker.patch(f"{_MODULE_PREFIX}.torch.nn.DataParallel")
        patched_parallel.return_value = mocker.MagicMock()
        model = mocker.MagicMock()
        conf = mod_base.TrainConfig(folders=["x", "y"],
                                    batch_size=batch_size,
                                    augment_color=False,
                                    flip=False,
                                    warp=False,
                                    cache_landmarks=False)
        instance = mod_distributed.Trainer(model, conf)
        return instance, patched_parallel

    return _apply_patch


@pytest.mark.parametrize("gpu_count", (2, 3, 5, 8))
@pytest.mark.parametrize("batch_size", (4, 8, 16, 32, 64))
def test_Trainer(gpu_count, batch_size, _trainer_mocked):
    """ Test that original trainer creates correctly """
    instance, patched_parallel = _trainer_mocked(gpus=gpu_count, batch_size=batch_size)
    assert isinstance(instance, mod_base.TrainerBase)
    assert isinstance(instance, mod_original.Trainer)
    # Confirms that _validate_batch_size executed correctly
    assert instance.batch_size == max(gpu_count, batch_size)
    assert hasattr(instance, "train_batch")
    # Confirms that _set_distributed executed correctly
    assert instance._distributed_model is patched_parallel.return_value


@pytest.mark.parametrize("gpu_count", (2, 3, 5, 8), ids=[f"gpus:{x}" for x in (2, 3, 5, 8)])
@pytest.mark.parametrize("outputs", (1, 2, 4))
@pytest.mark.parametrize("batch_size", (4, 8, 16, 32, 64))
def test_Trainer_forward(gpu_count, batch_size, outputs, _trainer_mocked, mocker):
    """ Test that original trainer _forward calls the correct model methods """
    instance, _ = _trainer_mocked(gpus=gpu_count, batch_size=batch_size)

    test_dims = (batch_size, 2, 16, 16, 3)

    inputs = list(torch.from_numpy(np.random.random(test_dims)).to("cpu"))
    targets = [torch.from_numpy(np.random.random(test_dims)).to("cpu")
               for _ in range(outputs)]

    loss_return = [DummyLoss() for _ in range(gpu_count)]
    instance._distributed_model = mocker.MagicMock(return_value=loss_return)
    instance._mean_loss = mocker.MagicMock(return_value={"unweighted": 1.0, "weighted": 1.0})

    # Call the forward pass
    instance._forward(inputs, targets, BatchMeta())

    # Make sure that our wrapped distributed model was called in the correct order
    instance._distributed_model.assert_called_once()
    call_args, call_kwargs = instance._distributed_model.call_args
    assert not call_kwargs
    assert len(call_args) == 3
