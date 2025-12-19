#!/usr/bin python3
""" Pytest unit tests for :mod:`plugins.train.trainer.distributed` Trainer plug in """
# pylint:disable=protected-access, invalid-name

import numpy as np
import pytest
import pytest_mock
import torch

from plugins.train.trainer import distributed as mod_distributed
from plugins.train.trainer import original as mod_original
from plugins.train.trainer import _base as mod_base


_MODULE_PREFIX = "plugins.train.trainer.distributed"


@pytest.mark.parametrize("batch_size", (4, 8, 16, 32, 64))
@pytest.mark.parametrize("outputs", (1, 2, 4))
def test_WrappedModel(batch_size, outputs, mocker):
    """ Test that the wrapped model calls preds and loss """
    model = mocker.MagicMock()
    instance = mod_distributed.WrappedModel(model)
    assert instance._keras_model is model

    loss_return = [torch.from_numpy((np.random.random((1, )))) for _ in range(outputs * 2)]
    model.loss = [mocker.MagicMock(return_value=ret) for ret in loss_return]

    test_dims = (batch_size, 16, 16, 3)

    inp_a = torch.from_numpy(np.random.random(test_dims))
    inp_b = torch.from_numpy(np.random.random(test_dims))
    targets = [torch.from_numpy(np.random.random(test_dims))
               for _ in range(outputs * 2)]
    preds = [*torch.from_numpy(np.random.random((outputs * 2, *test_dims)))]

    model.return_value = preds

    # Call forwards
    result = instance.forward(inp_a, inp_b, *targets)

    # Confirm model was called once forward with correct args
    model.assert_called_once()
    model_args, model_kwargs = model.call_args
    assert model_kwargs == {"training": True}
    assert len(model_args) == 1
    assert len(model_args[0]) == 2
    for real, expected in zip(model_args[0], [inp_a, inp_b]):
        assert np.allclose(real.numpy(), expected.numpy())

    # Confirm ZeroGrad called
    model.zero_grad.assert_called_once()

    # Confirm loss functions correctly called
    expected_targets = targets[0::2] + targets[1::2]

    for target, pred, loss in zip(expected_targets, preds, model.loss):
        loss.assert_called_once()
        loss_args, loss_kwargs = loss.call_args
        assert not loss_kwargs
        assert len(loss_args) == 2
        for actual, expected in zip(loss_args, [target, pred]):
            assert np.allclose(actual.numpy(), expected.numpy())

    # Check that the result comes out as we put it in
    for expected, actual in zip(loss_return, result.squeeze()):
        assert np.isclose(expected.numpy(), actual.numpy())


@pytest.fixture
def _trainer_mocked(mocker: pytest_mock.MockFixture):  # noqa:[F811]
    """ Generate a mocked model and feeder object and patch torch GPU count """

    def _apply_patch(gpus=2, batch_size=8):
        patched_cuda_device = mocker.patch(f"{_MODULE_PREFIX}.torch.cuda.device_count")
        patched_cuda_device.return_value = gpus
        patched_parallel = mocker.patch(f"{_MODULE_PREFIX}.torch.nn.DataParallel")
        patched_parallel.return_value = mocker.MagicMock()
        model = mocker.MagicMock()
        instance = mod_distributed.Trainer(model, batch_size)
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

    test_dims = (2, batch_size, 16, 16, 3)

    inputs = torch.from_numpy(np.random.random(test_dims))
    targets = [torch.from_numpy(np.random.random(test_dims)) for _ in range(outputs)]

    loss_return = torch.rand((gpu_count * 2 * outputs))
    instance._distributed_model = mocker.MagicMock(return_value=loss_return)

    # Call the forward pass
    result = instance._forward(inputs, targets).cpu().numpy()

    # Make sure multi-outs are enabled
    if outputs > 1:
        assert instance._is_multi_out is True
    else:
        assert instance._is_multi_out is False

    # Make sure that our wrapped distributed model was called in the correct order
    instance._distributed_model.assert_called_once()
    call_args, call_kwargs = instance._distributed_model.call_args
    assert not call_kwargs
    assert len(call_args) == len(inputs) + (len(targets) * 2)

    expected_tgts = [t[i].cpu().numpy() for t in targets for i in range(2)]

    for expected, actual in zip([*inputs, *expected_tgts], call_args):
        assert np.allclose(expected, actual)

    # Make sure loss gets grouped, summed and scaled correctly
    expected = loss_return.cpu().numpy()
    expected = expected.reshape((gpu_count, 2, -1)).sum(axis=0).flatten() / gpu_count
    assert np.allclose(result, expected)
